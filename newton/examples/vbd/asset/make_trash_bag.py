# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Procedural generator for a drawstring trash bag (VBD cloth asset).
#
# Builds a thin floppy plastic-bag shell whose rest shape is a flattened
# envelope: a rounded-rectangle cross-section (wide in x, shallow in y)
# extruded vertically, sealed at the bottom, open at the top. Along the
# flat FRONT (y=-b) and BACK (y=+b) panels the top hem is folded INWARD to
# form two channels (tunnels). The fold is kept MANIFOLD: the flap is an
# open folded flap (its free edge is a mesh boundary, NOT sewn back into the
# wall). The channel is closed at runtime by springs between each flap
# free-edge vertex and the wall vertex directly behind it (emitted to the
# layout JSON). The drawstring tie is a SEPARATE single-layer cloth ribbon
# (trash_bag_rope.obj) swept along a closed centerline that threads both
# tunnels and exits at the four flat/arc corners, forming two side handles you
# pull to cinch the bag shut.
#
# Outputs (all under this directory):
#   trash_bag.obj          - the bag cloth shell (wall + bottom cap + folds + flaps)
#   trash_bag_rope.obj     - the drawstring tie (single-layer cloth ribbon)
#   trash_bag_layout.json  - tunnel spring pairs, stripe band, drawstring centerline,
#                            rope ribbon info, 4 holes, params, counts
#
# Run:
#   python make_trash_bag.py                 # write obj + json + validate
#   python make_trash_bag.py --render        # also write annotated PNGs
###########################################################################

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Parameters (trash-can sized, SI metres, Z-up local frame z in [0, H]).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "W": 0.30,  # flat width   (x extent = 2a)
    "D": 0.08,  # depth        (y extent = 2b)  -> flattened (W/D = 3.75), the rest shape
    "H": 0.40,  # height       (z from 0 at bottom seam to H at fold/rim)
    "rc": 0.030,  # rounded-rect corner radius of the cross-section
    "fold": "out",  # "out" = hem folds outward (stripe band + handles on outside); "in" = inward
    "h_hem": 0.020,  # flap height = width of the folded hem/stripe band (kept narrow)
    "t_tunnel": 0.010,  # offset of the flap from the wall = channel (drawstring) thickness
    "fold_slope_drop": 0.005,  # top of flap drops below the rim so the fold slopes, not shelves
    "n_fold_slope": 2,  # divisions along the sloped fold before the vertical flap
    "ds": 0.012,  # target perimeter edge length (controls n around)
    "bottom_ds": 0.014,  # bottom-cap ring spacing (O-grid); ~matches perimeter ds
    "n_z": 40,  # vertical wall divisions (rows = n_z+1, plus fold-base row)
    "n_flap": 3,  # vertical flap divisions
    "ds_rope": 0.004,  # centerline segment length for the in-tunnel drawstring runs (handles resample themselves)
    "rope_width": 0.008,  # width of the single-layer cloth drawstring ribbon
    "rope_n_width": 2,  # segments across the ribbon width
    "rope_z_frac": 0.45,  # rope sits at z = H - rope_z_frac*h_hem (inside the channel)
    "rope_offset": 0.006,  # rope follows the bag contour, offset this far OUTWARD (no penetration)
    "handle_gap": 0.05,  # exposed handle gap at each side middle; tunnels cover the rest (incl. corners)
    "handle_stickout": 0.05,  # exposed handles bulge this far radially OUT of the holes (extra rope length)
    "handle_lift": 0.03,  # ...and rise this far, so the drawstring slack stands out above the rim
}


# ---------------------------------------------------------------------------
# Cross-section perimeter (rounded rectangle), CCW in the xy-plane.
# Each segment yields its points INCLUDING start, EXCLUDING end, so adjacent
# segments share their corner exactly once and the loop closes by wrap-around.
# ---------------------------------------------------------------------------
def build_perimeter(p):
    a, b, rc, ds = p["W"] / 2.0, p["D"] / 2.0, p["rc"], p["ds"]
    fx = a - rc  # front/back flat extends x in [-fx, +fx]
    sy = b - rc  # right/left straight extends y in [-sy, +sy]

    def n_for(length):
        return max(1, int(round(length / ds)))

    n_front = n_for(2.0 * fx)
    n_arc = max(2, n_for(0.5 * math.pi * rc))
    n_side = max(1, n_for(2.0 * sy))

    pts = []  # (x, y)
    labels = []  # per-vertex segment label

    def add(x, y, label):
        pts.append((float(x), float(y)))
        labels.append(label)

    # 1. front flat: x:-fx -> +fx at y=-b   (exclude +fx)
    for i in range(n_front):
        add(-fx + (i / n_front) * 2 * fx, -b, "front")
    # 2. front-right corner arc: center (fx,-sy), -90deg -> 0deg
    for i in range(n_arc):
        ang = math.radians(-90.0 + (i / n_arc) * 90.0)
        add(fx + rc * math.cos(ang), -sy + rc * math.sin(ang), "right")
    # 3. right straight: y:-sy -> +sy at x=+a
    for i in range(n_side):
        add(a, -sy + (i / n_side) * 2 * sy, "right")
    # 4. back-right corner arc: center (fx,+sy), 0deg -> 90deg
    for i in range(n_arc):
        ang = math.radians((i / n_arc) * 90.0)
        add(fx + rc * math.cos(ang), sy + rc * math.sin(ang), "right")
    # 5. back flat: x:+fx -> -fx at y=+b   (exclude -fx)
    for i in range(n_front):
        add(fx - (i / n_front) * 2 * fx, b, "back")
    # 6. back-left corner arc: center (-fx,+sy), 90deg -> 180deg
    for i in range(n_arc):
        ang = math.radians(90.0 + (i / n_arc) * 90.0)
        add(-fx + rc * math.cos(ang), sy + rc * math.sin(ang), "left")
    # 7. left straight: y:+sy -> -sy at x=-a
    for i in range(n_side):
        add(-a, sy - (i / n_side) * 2 * sy, "left")
    # 8. front-left corner arc: center (-fx,-sy), 180deg -> 270deg
    for i in range(n_arc):
        ang = math.radians(180.0 + (i / n_arc) * 90.0)
        add(-fx + rc * math.cos(ang), -sy + rc * math.sin(ang), "left")

    # index bookkeeping
    idx = {
        "n_front": n_front,
        "n_arc": n_arc,
        "n_side": n_side,
        "front_lo": 0,  # perimeter index of (-fx,-b)  (front-left exit)
        "front_hi": n_front,  # perimeter index of (+fx,-b)  (front-right exit)
        "back_lo": n_front + 2 * n_arc + n_side,  # (+fx,+b) back-right exit
        "back_hi": n_front + 2 * n_arc + n_side + n_front,  # (-fx,+b) back-left exit
    }
    return np.array(pts, dtype=np.float64), labels, idx


def _rr_sdf(x, y, a, b, rc):
    """Signed distance to the rounded-rectangle cross-section (<0 inside)."""
    qx = abs(x) - (a - rc)
    qy = abs(y) - (b - rc)
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - rc


def count_rope_penetrations(rope_verts, p, eps=1e-4):
    """Rope vertices that lie INSIDE the bag wall: inside the cross-section
    (sdf<0) and below the open rim (z<H). Those penetrate the bag."""
    a, b, rc, H = p["W"] / 2.0, p["D"] / 2.0, p["rc"], p["H"]
    n = 0
    for x, y, z in rope_verts:
        if z < H - eps and _rr_sdf(x, y, a, b, rc) < -eps:
            n += 1
    return n


def unflatten_to_cylinder(verts, p):
    """Map vertices of the flat (pressed-tube) rest bag onto a round cylinder of
    the SAME cross-section perimeter, so loop edge lengths are ~preserved (low
    membrane strain -> stable as a deformed INITIAL state over a flat rest shape).

    Each (x, y) is sent to polar coordinates: angle = its CCW arc-length fraction
    around the rounded-rect boundary, radius = R * (rho / rb(dir)) where R =
    perimeter / 2*pi and rb(dir) is the boundary radius in that direction. z is
    preserved. Works uniformly for wall (rho==rb -> R), bottom-cap interior
    (rho<rb -> inside the disk), folded flaps and the rope (rho>rb -> outside).
    """
    peri, _, _ = build_perimeter(p)  # ordered CCW boundary of the flat cross-section
    closed = np.vstack([peri, peri[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg)])  # arc-length param at each peri point (+closing)
    radius = total / (2.0 * math.pi)

    # boundary polar angle (unwrapped, monotonic since the convex section contains
    # the origin) and boundary radius, closed by appending the first point + 2*pi
    ang = np.unwrap(np.arctan2(peri[:, 1], peri[:, 0]))
    ang_closed = np.append(ang, ang[0] + 2.0 * math.pi)
    rb_closed = np.append(np.hypot(peri[:, 0], peri[:, 1]), np.hypot(peri[0, 0], peri[0, 1]))

    out = np.array(verts, dtype=np.float64).copy()
    for i in range(len(out)):
        x, y, z = out[i]
        rho = math.hypot(x, y)
        # bring the query direction into the unwrapped boundary range [ang[0], ang[0]+2pi)
        theta_q = ang[0] + math.fmod(math.atan2(y, x) - ang[0] + 2.0 * math.pi, 2.0 * math.pi)
        s = float(np.interp(theta_q, ang_closed, cum))
        rb_dir = float(np.interp(theta_q, ang_closed, rb_closed))
        phi = 2.0 * math.pi * (s / total)
        rr = radius * (rho / rb_dir) if rb_dir > 1e-9 else 0.0
        out[i] = [rr * math.cos(phi), rr * math.sin(phi), z]
    return out


def _outward_normals(peri):
    """Per-vertex outward (xy) unit normals for a CCW closed polygon."""
    tang = np.roll(peri, -1, axis=0) - np.roll(peri, 1, axis=0)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)  # tangent rotated -90deg = outward (CCW)
    return nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)


def _tunnel_runs(peri, gap_len):
    """Split a closed perimeter into the front/back TUNNEL runs and the two side
    handle GAPs. Each gap is centered on a side middle (+/- max|x|, y=0) and spans
    ~gap_len. Returns ordered contiguous (wrapping) index lists:
    (front_run, back_run, right_gap, left_gap)."""
    peri = np.asarray(peri)
    n = len(peri)
    a = float(np.max(np.abs(peri[:, 0])))
    half = 0.5 * gap_len
    dr = np.linalg.norm(peri - np.array([a, 0.0]), axis=1)
    dl = np.linalg.norm(peri - np.array([-a, 0.0]), axis=1)
    is_gap = (dr < half) | (dl < half)
    if not is_gap.any():
        return list(range(n)), [], [], []
    g0 = int(np.argmax(is_gap))  # start at a gap so non-gap runs come out contiguous
    runs, cur = [], []
    for k in range(n):
        i = (g0 + k) % n
        if is_gap[i]:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(i)
    if cur:
        runs.append(cur)
    front_run, back_run = [], []
    for run in runs:
        if float(np.mean(peri[run, 1])) < 0.0:
            front_run = run
        else:
            back_run = run
    right_gap = [i for i in range(n) if is_gap[i] and peri[i, 0] >= 0.0]
    left_gap = [i for i in range(n) if is_gap[i] and peri[i, 0] < 0.0]
    return front_run, back_run, right_gap, left_gap


def _rounded_rect_outline(fx, sy, rc, spacing):
    """Points around a rounded rectangle (core half-extents fx,sy; corner radius rc),
    sampled at ~`spacing` arc length, CCW. rc may be 0 (sharp rectangle). Each piece
    excludes its end point so consecutive pieces share corners without duplicates."""
    pts: list[tuple[float, float]] = []

    def line(p0, p1):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        n = max(1, int(round(np.linalg.norm(p1 - p0) / spacing)))
        for i in range(n):
            pts.append(tuple(p0 + (i / n) * (p1 - p0)))

    def arc(cx, cy, a0, a1):
        if rc <= 1e-9:
            return
        n = max(1, int(round(rc * abs(a1 - a0) / spacing)))
        for i in range(n):
            ang = a0 + (i / n) * (a1 - a0)
            pts.append((cx + rc * math.cos(ang), cy + rc * math.sin(ang)))

    line((-fx, -(sy + rc)), (fx, -(sy + rc)))
    arc(fx, -sy, -math.pi / 2, 0.0)
    line((fx + rc, -sy), (fx + rc, sy))
    arc(fx, sy, 0.0, math.pi / 2)
    line((fx, sy + rc), (-fx, sy + rc))
    arc(-fx, sy, math.pi / 2, math.pi)
    line((-fx - rc, sy), (-fx - rc, -sy))
    arc(-fx, -sy, math.pi, 1.5 * math.pi)
    return pts


# ---------------------------------------------------------------------------
# Bottom-cap triangulation. The cross-section is convex, so a Delaunay
# triangulation of (perimeter + interior points) fills it cleanly while keeping
# the exact boundary vertices (welded to the wall). The interior points are
# placed on boundary-parallel inset rings (an "O-grid") plus a central spine
# segment, so triangles run PARALLEL to the rim (no zig-zag) and there is no
# skinny centroid fan -- the shape collapses to the medial line, not a point.
# ---------------------------------------------------------------------------
def triangulate_bottom(peri, p):
    from scipy.spatial import Delaunay  # noqa: PLC0415  (build-time only)

    a, b, rc = p["W"] / 2.0, p["D"] / 2.0, p["rc"]
    h = p["bottom_ds"]
    fx0, sy0 = a - rc, b - rc  # core half-extents (invariant under inset while rc>0)

    interior: list[tuple[float, float]] = []
    d = h
    while True:
        if d < rc:
            fxr, syr, rcr = fx0, sy0, rc - d  # straight extents fixed, corner shrinks
        else:
            e = d - rc
            fxr, syr, rcr = fx0 - e, sy0 - e, 0.0  # sharp rect, shrinking
        if (b - d) < 0.7 * h or (fxr + rcr) < 0.7 * h:  # ring too thin -> stop, use spine
            break
        interior += _rounded_rect_outline(fxr, syr, rcr, h)
        d += h

    # central spine: the medial segment along x at y=0 (so the mesh collapses to a
    # line, not a point -> no skinny fan)
    x_sp = max(0.0, a - b)
    if x_sp > 0.3 * h:
        n = max(1, int(round(2 * x_sp / h)))
        interior += [(-x_sp + (i / n) * 2 * x_sp, 0.0) for i in range(n + 1)]
    else:
        interior += [(0.0, 0.0)]

    interior = np.array(interior, dtype=np.float64).reshape(-1, 2)
    pts = np.vstack([peri, interior]) if len(interior) else np.asarray(peri)
    return interior, Delaunay(pts).simplices


# ---------------------------------------------------------------------------
# Mesh assembly.
# ---------------------------------------------------------------------------
def build_mesh(p):
    H, h_hem, t = p["H"], p["h_hem"], p["t_tunnel"]
    b = p["D"] / 2.0

    peri, labels, idx = build_perimeter(p)
    P = len(peri)

    # Uniform wall rows. Rather than inserting an extra row at H-h_hem (which lands
    # next to a grid row and makes a ring of sliver triangles), SNAP the flap-attach
    # height to the nearest uniform row so the wall stays evenly spaced.
    zs = np.linspace(0.0, H, p["n_z"] + 1)
    k_top = len(zs) - 1  # top row at z=H (the fold line)
    k_fb = int(np.argmin(np.abs(zs - (H - h_hem))))  # nearest row to the requested fold base
    z_fold_base = float(zs[k_fb])
    h_hem = H - z_fold_base  # effective hem height (flap free edge sits on this wall row)

    verts: list[list[float]] = []

    def add_v(x, y, z):
        verts.append([float(x), float(y), float(z)])
        return len(verts) - 1

    # --- wall vertices: P columns x len(zs) rows ---
    wall = [[-1] * len(zs) for _ in range(P)]
    for pi in range(P):
        x, y = peri[pi]
        for k, z in enumerate(zs):
            wall[pi][k] = add_v(x, y, z)

    # --- bottom cap: quality Delaunay triangulation (perimeter + interior grid) ---
    bottom_interior, bottom_simplices = triangulate_bottom(peri, p)
    bottom_pts2d = np.vstack([peri, bottom_interior]) if len(bottom_interior) else np.asarray(peri)
    bottom_gmap = [wall[i][0] for i in range(P)] + [add_v(float(x), float(y), 0.0) for (x, y) in bottom_interior]

    # --- folded hem flaps, EXTENDED around the corners to the side ends; only a
    # small handle gap remains at each side middle ---
    fold_slope_drop = max(0.0, min(float(p.get("fold_slope_drop", 0.0)), h_hem - 1e-6))
    n_fold_slope = max(1, int(p.get("n_fold_slope", 1)))
    z_flap_top = H - fold_slope_drop
    flap_zs = np.linspace(z_flap_top, H - h_hem, p["n_flap"] + 1)[1:]
    fold_out = p.get("fold", "out") == "out"
    fold_sign = 1.0 if fold_out else -1.0  # +1 = fold outward (stripe band on the outside)
    normals = _outward_normals(peri)

    def build_flap(run):
        # flap columns offset off the wall along the local OUTWARD normal, so the hem
        # folds correctly around the corners (not just along +/-y)
        cols = []
        for pi in run:
            base = peri[pi] + fold_sign * t * normals[pi]
            col = []
            for step in range(1, n_fold_slope + 1):
                u = step / n_fold_slope
                fold_xy = peri[pi] + u * fold_sign * t * normals[pi]
                col.append(add_v(fold_xy[0], fold_xy[1], H - u * fold_slope_drop))
            col.extend(add_v(base[0], base[1], z) for z in flap_zs)
            cols.append(col)
        return cols

    front_cols_pidx, back_cols_pidx = _tunnel_runs(peri, p["handle_gap"])[:2]
    flap_front = build_flap(front_cols_pidx)
    flap_back = build_flap(back_cols_pidx)

    # --- faces (triangles, 0-indexed) ---
    faces: list[tuple[int, int, int]] = []

    def quad(v00, v10, v11, v01):
        # two triangles for quad with corners (v00,v10,v11,v01) wound consistently
        faces.append((v00, v10, v11))
        faces.append((v00, v11, v01))

    # wall grid (wrap around in perimeter)
    for pi in range(P):
        pn = (pi + 1) % P
        for k in range(len(zs) - 1):
            quad(wall[pi][k], wall[pn][k], wall[pn][k + 1], wall[pi][k + 1])

    # bottom cap (Delaunay tris, oriented so the normal faces down, -z)
    for s in bottom_simplices:
        g = [bottom_gmap[s[0]], bottom_gmap[s[1]], bottom_gmap[s[2]]]
        q0, q1, q2 = bottom_pts2d[s[0]], bottom_pts2d[s[1]], bottom_pts2d[s[2]]
        if (q1[0] - q0[0]) * (q2[1] - q0[1]) - (q2[0] - q0[0]) * (q1[1] - q0[1]) > 0.0:
            g[1], g[2] = g[2], g[1]
        faces.append((g[0], g[1], g[2]))

    # the folded hem band ("stripe") = fold-strip + flap faces; record their indices
    stripe_faces: list[int] = []

    def attach_flap(col_perimeter_indices, cols):
        start = len(faces)
        # fold strip: wall top row -> first sloped fold row
        for c in range(len(cols) - 1):
            pi = col_perimeter_indices[c]
            pn = col_perimeter_indices[c + 1]
            quad(wall[pi][k_top], wall[pn][k_top], cols[c + 1][0], cols[c][0])
        # remaining sloped rows and the vertical folded flap grid
        for c in range(len(cols) - 1):
            for k in range(len(cols[c]) - 1):
                quad(cols[c][k], cols[c + 1][k], cols[c + 1][k + 1], cols[c][k + 1])
        stripe_faces.extend(range(start, len(faces)))

    attach_flap(front_cols_pidx, flap_front)
    attach_flap(back_cols_pidx, flap_back)

    # --- tunnel-closure spring pairs: flap free edge <-> wall at fold-base row ---
    spring_pairs = []
    for c, pi in enumerate(front_cols_pidx):
        spring_pairs.append([int(flap_front[c][-1]), int(wall[pi][k_fb])])
    for c, pi in enumerate(back_cols_pidx):
        spring_pairs.append([int(flap_back[c][-1]), int(wall[pi][k_fb])])

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)

    meta = {
        "peri": peri,
        "labels": labels,
        "idx": idx,
        "zs": zs,
        "k_top": k_top,
        "k_fb": k_fb,
        "wall": wall,
        "front_cols_pidx": front_cols_pidx,
        "back_cols_pidx": back_cols_pidx,
        "flap_front": flap_front,
        "flap_back": flap_back,
        "spring_pairs": spring_pairs,
        "stripe_faces": stripe_faces,
        "fold_out": fold_out,
        "z_fold_base": z_fold_base,
        "z_flap_top": z_flap_top,
        "fold_slope_drop": fold_slope_drop,
        "n_fold_slope": n_fold_slope,
        "b": b,
    }
    return verts, faces, meta


def _resample_handle(base_xy, stickout, lift, rope_z, target_ds):
    """Build one exposed drawstring handle and resample it to ~square ribbon quads.

    ``base_xy`` are the flat contour points of the gap, in order. The handle bulges
    OUTWARD by ``stickout`` and UP by ``lift`` with a half-sine profile (zero at the
    holes, peak at the side middle), then is resampled by arc length so consecutive
    nodes sit ~``target_ds`` apart -- matching the ribbon's per-width spacing, so the
    stretched loop keeps ~square quads instead of long, skinny ones. Returns a list
    of ``[x, y, z]`` nodes; the endpoints stay unbulged and join the tunnel runs.
    """
    m = len(base_xy)
    if m <= 1:
        return [[float(x), float(y), rope_z] for (x, y) in base_xy]

    def bulged(t):
        f = t * (m - 1)
        i0 = min(int(f), m - 2)
        a = f - i0
        x = base_xy[i0][0] * (1 - a) + base_xy[i0 + 1][0] * a
        y = base_xy[i0][1] * (1 - a) + base_xy[i0 + 1][1] * a
        bump = math.sin(math.pi * t)
        rad = math.hypot(x, y)
        ux, uy = (x / rad, y / rad) if rad > 1e-9 else (1.0, 0.0)
        return [x + stickout * bump * ux, y + stickout * bump * uy, rope_z + lift * bump]

    # densely sample the bulged curve, measure its arc length, then resample evenly
    dense = np.array([bulged(j / 400.0) for j in range(401)])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    length = float(cum[-1])
    n_seg = max(m - 1, int(round(length / target_ds)))
    ss = np.linspace(0.0, length, n_seg + 1)
    xs, ys, zs = (np.interp(ss, cum, dense[:, c]) for c in range(3))
    return [[float(x), float(y), float(z)] for x, y, z in zip(xs, ys, zs, strict=True)]


# ---------------------------------------------------------------------------
# Drawstring centerline = the bag's perimeter contour OFFSET OUTWARD by a small
# amount, at z = rope_z. An outward offset of a rounded rectangle is the same
# shape with corner radius rc+offset (core half-extents unchanged), so we reuse
# build_perimeter with enlarged params -> the rope is guaranteed `offset` outside
# the wall everywhere (no penetration). The front/back flats run inside the hem
# tunnels; the exposed left/right rounded ends are the two handles.
# ---------------------------------------------------------------------------
def build_drawstring(p, mesh_meta):
    a = p["W"] / 2.0
    H, h_hem = p["H"], p["h_hem"]
    rope_z = H - p["rope_z_frac"] * h_hem
    offset = p["rope_offset"]

    # offset contour: same flats (fx, sy), corner radius rc+offset
    pp = dict(p)
    pp["W"] = p["W"] + 2.0 * offset
    pp["D"] = p["D"] + 2.0 * offset
    pp["rc"] = p["rc"] + offset
    pp["ds"] = p["ds_rope"]
    cxy, _clabels, _cidx = build_perimeter(pp)
    n = len(cxy)

    # tunnels extend around the corners; only the small side-middle gaps are exposed
    _front_run, back_run, right_gap, left_gap = _tunnel_runs(cxy, p["handle_gap"])
    kind = ["front_tunnel"] * n
    for i in back_run:
        kind[i] = "back_tunnel"
    for i in right_gap:
        kind[i] = "right_handle"
    for i in left_gap:
        kind[i] = "left_handle"

    # Each exposed handle bulges OUTWARD (and up) into a grab loop whose arc length is
    # several times the flat gap it spans, so the slack stands out of the side holes.
    # Sweeping the ribbon over the original (coarse) gap nodes would leave long, skinny
    # quads there; instead resample each bulged handle at the ribbon's per-width
    # spacing so its quads stay ~square. Tunnel runs keep the coarse ds_rope spacing.
    stickout = p.get("handle_stickout", 0.0)
    lift = p.get("handle_lift", 0.0)
    target_ds = p["rope_width"] / p["rope_n_width"]

    # walk the contour in order: keep tunnel nodes as-is, replace each contiguous
    # handle gap with its resampled bulged loop (gaps do not wrap the seam)
    path: list[list[float]] = []
    seg: list[str] = []
    handle_node_indices = {"right": [], "left": []}
    i = 0
    while i < n:
        k = kind[i]
        if k in ("right_handle", "left_handle"):
            j = i
            while j < n and kind[j] == k:
                j += 1
            loop = _resample_handle([cxy[t] for t in range(i, j)], stickout, lift, rope_z, target_ds)
            side = "right" if k == "right_handle" else "left"
            handle_node_indices[side] = list(range(len(path), len(path) + len(loop)))
            path.extend(loop)
            seg.extend([k] * len(loop))
            i = j
        else:
            path.append([float(cxy[i][0]), float(cxy[i][1]), rope_z])
            seg.append(k)
            i += 1

    drawstring = {
        "closed": True,
        "rope_width": p["rope_width"],
        "rope_offset": offset,
        "handle_gap": p["handle_gap"],
        "rope_z": rope_z,
        "n_nodes": len(path),
        "path": path,
        "labels": seg,
        # rope vertices forming each exposed handle (the side-middle gaps):
        "handle_node_indices": handle_node_indices,
    }
    # the two exposed exit regions (side middles), where the drawstring leaves the tunnels
    holes = {
        "right": [a, 0.0, rope_z],
        "left": [-a, 0.0, rope_z],
    }
    return drawstring, holes


# ---------------------------------------------------------------------------
# Drawstring as a thin SINGLE-LAYER cloth ribbon (its own OBJ) -- NOT a tube.
# Sweep a short line segment (the ribbon width) along the closed centerline.
# The path is planar (z = rope_z), so a twist-free frame is tangent x world-up;
# the width runs along the (near-vertical) normal so the ribbon stands in the
# channel, facing the wall/flap. Closed loop => an open band (cylinder topology:
# 2 long boundary edges, no caps) -> a single sheet of cloth. Vertices ordered
# row-by-row: vertex(i, k) = i*(n_width+1) + k; segment label = labels[i].
# ---------------------------------------------------------------------------
def build_rope_strip(drawstring, p):
    path = np.array(drawstring["path"], dtype=np.float64)
    labels = drawstring["labels"]
    n = len(path)
    w = p["rope_width"]
    nw = p["rope_n_width"]
    up = np.array([0.0, 0.0, 1.0])

    verts = []
    vlabels = []
    rows = []
    for i in range(n):
        tang = path[(i + 1) % n] - path[(i - 1) % n]
        tn = np.linalg.norm(tang)
        tang = tang / tn if tn > 1e-12 else np.array([1.0, 0.0, 0.0])
        binormal = np.cross(tang, up)
        bn = np.linalg.norm(binormal)
        binormal = binormal / bn if bn > 1e-9 else np.array([1.0, 0.0, 0.0])
        normal = np.cross(binormal, tang)  # ~ world up, perpendicular to tangent
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        row = []
        for k in range(nw + 1):
            s = (k / nw - 0.5) * w  # -w/2 .. +w/2 across the ribbon width
            row.append(len(verts))
            verts.append((path[i] + s * normal).tolist())
            vlabels.append(labels[i])
        rows.append(row)

    faces = []
    for i in range(n):
        j = (i + 1) % n
        for k in range(nw):
            v00, v10, v11, v01 = rows[i][k], rows[i][k + 1], rows[j][k + 1], rows[j][k]
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64), vlabels


# ---------------------------------------------------------------------------
# OBJ writer (triangles, 1-indexed).
# ---------------------------------------------------------------------------
def write_obj(path, verts, faces):
    lines = ["# drawstring trash bag - generated by make_trash_bag.py\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    with open(path, "w") as fh:
        fh.writelines(lines)


# ---------------------------------------------------------------------------
# Validation: manifold (no edge with >2 faces), single connected component,
# no degenerate faces, no duplicate vertices.
# ---------------------------------------------------------------------------
def validate(verts, faces):
    report = {}
    nv, nf = len(verts), len(faces)
    report["num_vertices"] = nv
    report["num_faces"] = nf

    # degenerate faces (zero area)
    v = verts
    a = v[faces[:, 1]] - v[faces[:, 0]]
    bb = v[faces[:, 2]] - v[faces[:, 0]]
    cross = np.cross(a, bb)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    report["min_face_area"] = float(areas.min())
    report["num_degenerate_faces"] = int((areas < 1e-12).sum())

    # edge -> face count
    edge_count = {}
    for f in faces:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (int(min(e)), int(max(e)))
            edge_count[key] = edge_count.get(key, 0) + 1
    counts = np.array(list(edge_count.values()))
    report["num_edges"] = int(len(edge_count))
    report["max_faces_per_edge"] = int(counts.max())
    report["num_nonmanifold_edges"] = int((counts > 2).sum())
    report["num_boundary_edges"] = int((counts == 1).sum())
    report["is_manifold"] = bool((counts <= 2).all())

    # duplicate vertices (rounded)
    keyset = {}
    dup = 0
    for i, vi in enumerate(verts):
        k = (round(vi[0], 7), round(vi[1], 7), round(vi[2], 7))
        if k in keyset:
            dup += 1
        else:
            keyset[k] = i
    report["num_duplicate_vertices"] = int(dup)

    # connected components via union-find over faces
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for f in faces:
        union(int(f[0]), int(f[1]))
        union(int(f[1]), int(f[2]))
    roots = {find(i) for i in range(nv)}
    report["num_connected_components"] = int(len(roots))

    # cross-check with trimesh if available
    try:
        import trimesh  # noqa: PLC0415  (optional cross-check dependency)

        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report["trimesh_is_winding_consistent"] = bool(m.is_winding_consistent)
        report["trimesh_body_count"] = int(m.body_count)
        report["trimesh_euler_number"] = int(m.euler_number)
    except Exception as e:  # pragma: no cover
        report["trimesh"] = f"unavailable: {e}"

    report["bbox_min"] = verts.min(axis=0).tolist()
    report["bbox_max"] = verts.max(axis=0).tolist()
    return report


# ---------------------------------------------------------------------------
# Optional annotated render (matplotlib).
# ---------------------------------------------------------------------------
def render(verts, faces, layout, out_prefix, p, rope_verts=None, rope_faces=None):
    import matplotlib  # noqa: PLC0415  (render-only deps, kept lazy)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    ds = np.array(layout["drawstring"]["path"])
    seg = layout["drawstring"]["labels"]
    holes = layout["holes"]
    seg_colors = {
        "front_tunnel": "#1f77b4",
        "back_tunnel": "#2ca02c",
        "right_handle": "#d62728",
        "left_handle": "#9467bd",
    }

    def draw_rope(ax, three_d=True):
        path = ds
        n = len(path)
        for i in range(n):
            j = (i + 1) % n
            c = seg_colors.get(seg[i], "#000")
            if three_d:
                ax.plot([path[i, 0], path[j, 0]], [path[i, 1], path[j, 1]], [path[i, 2], path[j, 2]], color=c, lw=2.5)
            else:
                ax.plot([path[i, 0], path[j, 0]], [path[i, 1], path[j, 1]], color=c, lw=2.5)

    tris = verts[faces]
    stripe_set = set(layout["stripe"]["face_indices"])
    stripe_mask = np.array([i in stripe_set for i in range(len(faces))])
    fold_out = layout["stripe"]["fold"] == "out"
    CLOTH = "#9bb6d6"
    STRIPE = "#f0820a"  # the colored drawstring stripe band

    def add_cloth(ax, mask=None):
        sel = np.ones(len(faces), bool) if mask is None else mask
        ax.add_collection3d(
            Poly3DCollection(
                tris[sel & ~stripe_mask], alpha=0.20, facecolor=CLOTH, edgecolor="#6a86a8", linewidths=0.08
            )
        )
        ax.add_collection3d(
            Poly3DCollection(tris[sel & stripe_mask], alpha=0.95, facecolor=STRIPE, edgecolor="#9a5200", linewidths=0.2)
        )

    # ---- view 1: oblique 3D (full bag) ----
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    add_cloth(ax)
    draw_rope(ax, three_d=True)
    for hp in holes.values():
        ax.scatter([hp[0]], [hp[1]], [hp[2]], color="k", s=45, marker="o")
    ax.set_title(
        "Drawstring trash bag — full shell\n(orange = folded-out hem stripe, colored lines = drawstring, dots = 4 holes)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_axes_equal(ax, verts)
    ax.set_box_aspect((p["W"], p["D"], p["H"]))
    ax.view_init(elev=16, azim=-72)
    fig.savefig(f"{out_prefix}_oblique.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- view 1b: 3D close-up of the folded top (stripe band + drawstring) ----
    z_cut = p["H"] - 3.0 * p["h_hem"]
    keep = verts[faces][:, :, 2].max(axis=1) > z_cut
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    add_cloth(ax, keep)
    draw_rope(ax, three_d=True)
    for name, hp in holes.items():
        ax.scatter([hp[0]], [hp[1]], [hp[2]], color="k", s=55, marker="o")
        ax.text(hp[0], hp[1], hp[2] + 0.006, name.replace("_", "-"), fontsize=8)
    ax.text(0, -p["D"] / 2, p["H"] + 0.012, "FRONT", color=seg_colors["front_tunnel"], fontsize=11, ha="center")
    ax.text(0, p["D"] / 2, p["H"] + 0.012, "BACK", color=seg_colors["back_tunnel"], fontsize=11, ha="center")
    ax.set_title(
        "Top close-up: outward-folded hem (orange stripe) forms the two tunnels;\ndrawstring threads both and exits the 4 holes as 2 side handles"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-p["W"] / 2 - 0.03, p["W"] / 2 + 0.03)
    ax.set_ylim(-p["D"] / 2 - 0.06, p["D"] / 2 + 0.06)
    ax.set_zlim(z_cut, p["H"] + 0.02)
    ax.set_box_aspect((p["W"] + 0.06, p["D"] + 0.12, (p["H"] - z_cut) * 2.0))
    ax.view_init(elev=24, azim=-66)
    fig.savefig(f"{out_prefix}_top_closeup.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- view 2: top-down (xy) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, tri in enumerate(tris):
        xy = tri[:, :2]
        if stripe_mask[i]:
            ax.fill(xy[:, 0], xy[:, 1], color=STRIPE, alpha=0.22, lw=0)
        else:
            ax.fill(xy[:, 0], xy[:, 1], color="#ccc", alpha=0.03, lw=0)
    draw_rope(ax, three_d=False)
    for name, hp in holes.items():
        ax.scatter([hp[0]], [hp[1]], color="k", s=45, zorder=5)
        ax.annotate(name.replace("_", "-"), (hp[0], hp[1]), fontsize=8, textcoords="offset points", xytext=(4, 4))
    ax.text(
        0,
        -p["D"] / 2 - 0.025,
        "FRONT tunnel (in orange stripe)",
        color=seg_colors["front_tunnel"],
        ha="center",
        fontsize=9,
    )
    ax.text(
        0,
        p["D"] / 2 + 0.018,
        "BACK tunnel (in orange stripe)",
        color=seg_colors["back_tunnel"],
        ha="center",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_title("Top-down (xy): orange stripe = the two tunnels (front/back); 4 holes; drawstring loop + side handles")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    fig.savefig(f"{out_prefix}_top.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- view 3: schematic y-z cross-section of the folded hem ----
    fig, ax = plt.subplots(figsize=(7, 7))
    b = p["D"] / 2
    H, h_hem, t = p["H"], p["h_hem"], p["t_tunnel"]
    zfb = H - h_hem
    rz = layout["drawstring"]["rope_z"]
    sf = -1.0 if fold_out else +1.0  # front flap offset sign (out = -y)
    sb = +1.0 if fold_out else -1.0  # back flap offset sign
    flap_lbl = "outward-folded hem (stripe band)" if fold_out else "inward-folded hem (stripe band)"
    # FRONT side (wall at y=-b, flap at y=-b+sf*t)
    ax.plot([-b, -b], [zfb - 0.03, H], color="#333", lw=3, label="bag wall")
    ax.plot([-b, -b + sf * t], [H, H], color=STRIPE, lw=4)  # fold cap (top)
    ax.plot([-b + sf * t, -b + sf * t], [H, zfb], color=STRIPE, lw=4, label=flap_lbl)
    ax.plot([-b + sf * t, -b], [zfb, zfb], color="#1f77b4", lw=2, ls="--", label="closure spring (flap<->wall)")
    ax.plot(
        [-b + sf * 0.5 * t, -b + sf * 0.5 * t],
        [rz - p["rope_width"] / 2, rz + p["rope_width"] / 2],
        color=seg_colors["front_tunnel"],
        lw=4,
        solid_capstyle="butt",
    )  # single-layer ribbon (edge-on)
    ax.annotate(
        "drawstring",
        (-b + sf * 0.5 * t, rz),
        textcoords="offset points",
        xytext=(-10 if fold_out else 10, 0),
        ha="right" if fold_out else "left",
        fontsize=9,
    )
    # BACK side mirror
    ax.plot([b, b], [zfb - 0.03, H], color="#333", lw=3)
    ax.plot([b, b + sb * t], [H, H], color=STRIPE, lw=4)
    ax.plot([b + sb * t, b + sb * t], [H, zfb], color=STRIPE, lw=4)
    ax.plot([b + sb * t, b], [zfb, zfb], color="#1f77b4", lw=2, ls="--")
    ax.plot(
        [b + sb * 0.5 * t, b + sb * 0.5 * t],
        [rz - p["rope_width"] / 2, rz + p["rope_width"] / 2],
        color=seg_colors["back_tunnel"],
        lw=4,
        solid_capstyle="butt",
    )  # single-layer ribbon (edge-on)
    ax.annotate("fold line (rim, z=H)", (0, H), ha="center", fontsize=9, textcoords="offset points", xytext=(0, 6))
    ax.set_aspect("equal")
    ax.set_xlabel("y  (depth)")
    ax.set_ylabel("z  (height)")
    ax.set_title(
        "Folded-hem cross-section (y-z): manifold flap + closure springs form each tunnel\n(hem folds OUTWARD = stripe on the outside)"
    )
    ax.legend(loc="lower center", fontsize=8)
    ax.set_xlim(-b - t - 0.02, b + t + 0.02)
    ax.set_ylim(zfb - 0.04, H + 0.03)
    fig.savefig(f"{out_prefix}_hem_section.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- view 4: bottom-cap tessellation (xy wireframe) ----
    zc = verts[:, 2]
    cap_mask = (zc[faces] < 1e-6).all(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in faces[cap_mask]:
        loop = np.vstack([verts[f][:, :2], verts[f][0, :2]])
        ax.plot(loop[:, 0], loop[:, 1], color="#3f5d80", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"Bottom cap: Delaunay tessellation ({int(cap_mask.sum())} triangles) — even sizing, no skinny fan")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(f"{out_prefix}_bottom.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    out = [
        f"{out_prefix}_oblique.png",
        f"{out_prefix}_top_closeup.png",
        f"{out_prefix}_top.png",
        f"{out_prefix}_hem_section.png",
        f"{out_prefix}_bottom.png",
    ]

    # ---- view 5: the drawstring tie as its own single-layer cloth ribbon ----
    if rope_verts is not None and rope_faces is not None:
        rtris = rope_verts[rope_faces]
        # (a) ribbon alone
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(rtris, alpha=0.85, facecolor="#d62728", edgecolor="#5a0000", linewidths=0.3)
        )
        ax.set_title(
            f"Drawstring tie — single-layer cloth ribbon (separate OBJ)\n{len(rope_verts)} verts / {len(rope_faces)} tris, width {p['rope_width']} m"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        _set_axes_equal(ax, rope_verts)
        ax.view_init(elev=38, azim=-62)
        fig.savefig(f"{out_prefix}_rope.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        out.append(f"{out_prefix}_rope.png")

        # (b) ribbon + bag together (bag translucent)
        fig = plt.figure(figsize=(8, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris[~stripe_mask], alpha=0.10, facecolor=CLOTH, edgecolor="none"))
        ax.add_collection3d(Poly3DCollection(tris[stripe_mask], alpha=0.30, facecolor=STRIPE, edgecolor="none"))
        ax.add_collection3d(
            Poly3DCollection(rtris, alpha=0.95, facecolor="#d62728", edgecolor="#5a0000", linewidths=0.2)
        )
        ax.set_title("Drawstring tie (red ribbon) threaded through the bag's stripe band + handles")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        _set_axes_equal(ax, verts)
        ax.set_box_aspect((p["W"], p["D"], p["H"]))
        ax.view_init(elev=16, azim=-72)
        fig.savefig(f"{out_prefix}_rope_in_bag.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        out.append(f"{out_prefix}_rope_in_bag.png")

    return out


def _set_axes_equal(ax, verts):
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    r = 0.5 * (maxs - mins).max()
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="also write annotated PNGs")
    ap.add_argument("--out-dir", default=HERE)
    for k, val in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(val), default=val)
    args = ap.parse_args()
    p = {k: getattr(args, k) for k in DEFAULTS}

    verts, faces, mesh_meta = build_mesh(p)
    drawstring, holes = build_drawstring(p, mesh_meta)
    report = validate(verts, faces)

    # drawstring tie = its own single-layer cloth ribbon (separate OBJ)
    rope_verts, rope_faces, rope_vlabels = build_rope_strip(drawstring, p)
    rope_report = validate(rope_verts, rope_faces)
    right_handle_v = [i for i, lbl in enumerate(rope_vlabels) if lbl == "right_handle"]
    left_handle_v = [i for i, lbl in enumerate(rope_vlabels) if lbl == "left_handle"]

    obj_path = os.path.join(args.out_dir, "trash_bag.obj")
    rope_obj_path = os.path.join(args.out_dir, "trash_bag_rope.obj")
    json_path = os.path.join(args.out_dir, "trash_bag_layout.json")
    write_obj(obj_path, verts, faces)
    write_obj(rope_obj_path, rope_verts, rope_faces)

    # round INITIAL positions (same topology) = the flat rest bag unflattened onto
    # a cylinder, so the demo can start the bag round (e.g. lining a round bin)
    # while its rest shape stays the flat pressed tube. The example loads these as
    # the starting particle_q and builds the cloth rest state from the flat OBJ.
    verts_init = unflatten_to_cylinder(verts, p)
    rope_verts_init = unflatten_to_cylinder(rope_verts, p)
    write_obj(os.path.join(args.out_dir, "trash_bag_init.obj"), verts_init, faces)
    write_obj(os.path.join(args.out_dir, "trash_bag_rope_init.obj"), rope_verts_init, rope_faces)

    layout = {
        "description": "Drawstring trash bag layout. Vertex indices are 0-based into "
        "trash_bag.obj IN FILE ORDER. Load the OBJ with process=False to preserve order.",
        "params": p,
        "counts": {
            "num_vertices": int(len(verts)),
            "num_faces": int(len(faces)),
            "num_tunnel_springs": int(len(mesh_meta["spring_pairs"])),
            "num_stripe_faces": int(len(mesh_meta["stripe_faces"])),
        },
        "frame": "Z-up, local; bag bottom seam at z=0, fold/rim at z=H. Apply the same pos "
        "offset to both the cloth mesh and this layout in the demo.",
        # tunnel-closure springs: each [flap_free_edge_vertex, wall_vertex_at_fold_base]
        "tunnel_spring_pairs": mesh_meta["spring_pairs"],
        # the folded hem band = the classic colored drawstring "stripe" (fold is OUTward)
        "stripe": {
            "description": "Folded-over hem band (the colored drawstring stripe). Indices into "
            "trash_bag.obj; color these faces distinctly to render the stripe.",
            "fold": "out" if mesh_meta["fold_out"] else "in",
            "face_indices": [int(f) for f in mesh_meta["stripe_faces"]],
            "vertex_indices": sorted({int(v) for f in mesh_meta["stripe_faces"] for v in faces[f]}),
        },
        "drawstring": drawstring,
        # the tie itself = a separate single-layer cloth ribbon (its own OBJ)
        "rope": {
            "obj": "trash_bag_rope.obj",
            "description": "Single-layer cloth ribbon = the drawstring tie (a separate cloth mesh, "
            "NOT a tube). Load with process=False. Vertex(i,k) = i*(n_width+1)+k along the centerline.",
            "width": p["rope_width"],
            "n_width": p["rope_n_width"],
            "n_centerline": len(drawstring["path"]),
            "num_vertices": int(len(rope_verts)),
            "num_faces": int(len(rope_faces)),
            "vertex_labels": rope_vlabels,
            # rope vertices forming each exposed handle (pin/pull these to cinch):
            "handle_vertex_indices": {"right": right_handle_v, "left": left_handle_v},
            "validation": rope_report,
        },
        "holes": holes,
        "validation": report,
    }
    with open(json_path, "w") as fh:
        json.dump(layout, fh, indent=2)

    print("=" * 70)
    print(f"wrote {obj_path}")
    print(f"wrote {rope_obj_path}")
    print(f"wrote {json_path}")
    print("-" * 70)
    for k in [
        "num_vertices",
        "num_faces",
        "num_edges",
        "num_boundary_edges",
        "max_faces_per_edge",
        "num_nonmanifold_edges",
        "is_manifold",
        "num_degenerate_faces",
        "min_face_area",
        "num_duplicate_vertices",
        "num_connected_components",
        "trimesh_is_winding_consistent",
        "trimesh_body_count",
        "bbox_min",
        "bbox_max",
    ]:
        if k in report:
            print(f"  {k:28s}: {report[k]}")
    print(f"  num_tunnel_springs          : {len(mesh_meta['spring_pairs'])}")
    print(f"  drawstring_nodes            : {drawstring['n_nodes']}")
    print(f"  rope(ribbon) verts/faces    : {len(rope_verts)} / {len(rope_faces)}")
    print(
        f"  rope is_manifold            : {rope_report['is_manifold']} "
        f"(boundary edges={rope_report['num_boundary_edges']}, components={rope_report['num_connected_components']})"
    )
    print(f"  rope penetrations into bag  : {count_rope_penetrations(rope_verts, p)} (want 0)")
    print("=" * 70)

    if args.render:
        pngs = render(verts, faces, layout, os.path.join(args.out_dir, "trash_bag"), p, rope_verts, rope_faces)
        print("rendered:")
        for pp in pngs:
            print("  ", pp)


if __name__ == "__main__":
    main()
