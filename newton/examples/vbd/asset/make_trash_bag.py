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
# layout JSON). A drawstring rope (particles+springs, built in the demo
# script) threads both tunnels and exits at the four flat/arc corners,
# forming two side handles you can pull to cinch the bag shut.
#
# Outputs (all under this directory):
#   trash_bag.obj          - the cloth mesh (wall + bottom cap + folds + flaps)
#   trash_bag_layout.json  - tunnel spring pairs, ordered drawstring path,
#                            4 exit/hole locations, handle spans, params, counts
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
    "D": 0.12,  # depth        (y extent = 2b)  -> flattened (W/D = 2.5)
    "H": 0.40,  # height       (z from 0 at bottom seam to H at fold/rim)
    "rc": 0.045,  # rounded-rect corner radius of the cross-section
    "fold": "out",  # "out" = hem folds outward (stripe band + handles on outside); "in" = inward
    "h_hem": 0.028,  # flap height = width of the folded hem/stripe band (kept narrow)
    "t_tunnel": 0.014,  # offset of the flap from the wall = channel (drawstring) thickness
    "ds": 0.012,  # target perimeter edge length (controls n around)
    "n_z": 28,  # vertical wall divisions (rows = n_z+1, plus fold-base row)
    "n_flap": 5,  # vertical flap divisions
    "ds_rope": 0.018,  # target drawstring segment length
    "rope_radius": 0.005,
    "rope_z_frac": 0.45,  # rope sits at z = H - rope_z_frac*h_hem (inside the channel)
    "handle_bulge": 0.03,  # how far (m) the handle arc bows outward past the side rim
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


# ---------------------------------------------------------------------------
# Mesh assembly.
# ---------------------------------------------------------------------------
def build_mesh(p):
    H, h_hem, t = p["H"], p["h_hem"], p["t_tunnel"]
    b = p["D"] / 2.0

    peri, labels, idx = build_perimeter(p)
    P = len(peri)

    # z levels for the wall: uniform + guarantee a row exactly at the fold base.
    z_fold_base = H - h_hem
    zs = sorted({*np.linspace(0.0, H, p["n_z"] + 1).tolist(), z_fold_base})
    zs = np.array(zs, dtype=np.float64)
    k_top = int(np.argmin(np.abs(zs - H)))  # row at the fold line z=H
    k_fb = int(np.argmin(np.abs(zs - z_fold_base)))  # row at z=H-h_hem

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

    # --- bottom cap centroid ---
    center_b = add_v(0.0, 0.0, 0.0)

    # --- flap columns along the front flat and back flat (inward hem) ---
    flap_zs = np.linspace(H, H - h_hem, p["n_flap"] + 1)  # k=0 at fold, last at free edge

    # Fold direction: "out" puts the hem/stripe band + handles on the OUTSIDE
    # (front wall at y=-b folds toward -y); "in" folds toward the interior.
    fold_out = p.get("fold", "out") == "out"
    front_sign = -1.0 if fold_out else +1.0  # offset of +t off the front wall (y=-b)
    back_sign = +1.0 if fold_out else -1.0  # offset of +t off the back wall  (y=+b)

    def build_flap(col_perimeter_indices, sign):
        # flap column offset off the wall by sign*t (sign chooses the in/out side)
        cols = []
        for pi in col_perimeter_indices:
            x = peri[pi][0]
            y_flap = peri[pi][1] + sign * t
            col = [add_v(x, y_flap, z) for z in flap_zs]
            cols.append(col)
        return cols

    front_cols_pidx = list(range(idx["front_lo"], idx["front_hi"] + 1))
    back_cols_pidx = list(range(idx["back_lo"], idx["back_hi"] + 1))
    flap_front = build_flap(front_cols_pidx, front_sign)
    flap_back = build_flap(back_cols_pidx, back_sign)

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

    # bottom cap fan (normal pointing down): center, next, this
    for pi in range(P):
        pn = (pi + 1) % P
        faces.append((center_b, wall[pn][0], wall[pi][0]))

    # the folded hem band ("stripe") = fold-strip + flap faces; record their indices
    stripe_faces: list[int] = []

    def attach_flap(col_perimeter_indices, cols):
        start = len(faces)
        # fold strip: wall top row -> flap top row
        for c in range(len(cols) - 1):
            pi = col_perimeter_indices[c]
            pn = col_perimeter_indices[c + 1]
            quad(wall[pi][k_top], wall[pn][k_top], cols[c + 1][0], cols[c][0])
        # flap patch grid
        for c in range(len(cols) - 1):
            for k in range(len(flap_zs) - 1):
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
        "b": b,
    }
    return verts, faces, meta


# ---------------------------------------------------------------------------
# Drawstring path (closed loop): front tunnel -> right handle -> back tunnel
# -> left handle -> close. Emitted to JSON; particles/springs built in Step 2.
# ---------------------------------------------------------------------------
def build_drawstring(p, mesh_meta):
    a, b = p["W"] / 2.0, p["D"] / 2.0
    fx = a - p["rc"]
    t = p["t_tunnel"]
    H, h_hem = p["H"], p["h_hem"]
    rope_z = H - p["rope_z_frac"] * h_hem
    fold_out = p.get("fold", "out") == "out"
    front_sign = -1.0 if fold_out else +1.0
    back_sign = +1.0 if fold_out else -1.0
    y_front = -b + front_sign * 0.5 * t  # mid-channel on the folded side
    y_back = b + back_sign * 0.5 * t

    n_tun = max(2, int(round(2 * fx / p["ds_rope"])))

    path = []  # list of [x,y,z]
    seg = []  # per-node label

    def add(x, y, z, label):
        path.append([float(x), float(y), float(z)])
        seg.append(label)

    # front tunnel: x:-fx -> +fx (inclusive both ends)
    for i in range(n_tun + 1):
        x = -fx + (i / n_tun) * 2 * fx
        add(x, y_front, rope_z, "front_tunnel")
    front_lo_node = 0
    front_hi_node = len(path) - 1

    # right handle: quadratic Bezier A=(+fx,y_front) -> B=(+fx,y_back), bow out +x
    A = np.array([fx, y_front, rope_z])
    B = np.array([fx, y_back, rope_z])
    Cr = np.array([a + p["handle_bulge"], 0.0, rope_z])
    n_handle = 6
    for i in range(1, n_handle):  # interior nodes only (A,B already in tunnels)
        u = i / n_handle
        pt = (1 - u) ** 2 * A + 2 * (1 - u) * u * Cr + u**2 * B
        add(pt[0], pt[1], pt[2], "right_handle")

    # back tunnel: x:+fx -> -fx (inclusive both ends)
    for i in range(n_tun + 1):
        x = fx - (i / n_tun) * 2 * fx
        add(x, y_back, rope_z, "back_tunnel")
    back_lo_node = front_hi_node + (n_handle - 1) + 1
    back_hi_node = len(path) - 1

    # left handle: Bezier from back-left -> front-left, bow out -x
    A2 = np.array([-fx, y_back, rope_z])
    B2 = np.array([-fx, y_front, rope_z])
    Cl = np.array([-a - p["handle_bulge"], 0.0, rope_z])
    for i in range(1, n_handle):
        u = i / n_handle
        pt = (1 - u) ** 2 * A2 + 2 * (1 - u) * u * Cl + u**2 * B2
        add(pt[0], pt[1], pt[2], "left_handle")

    drawstring = {
        "closed": True,
        "rope_radius_hint": p["rope_radius"],
        "rope_z": rope_z,
        "n_nodes": len(path),
        "path": path,
        "labels": seg,
        # the four exit/hole nodes on the rope (where it leaves a channel):
        "exit_rope_nodes": {
            "front_left": front_lo_node,
            "front_right": front_hi_node,
            "back_right": back_lo_node,
            "back_left": back_hi_node,
        },
        # rope index ranges that form the two exposed handles (to pull/cinch):
        "handle_spans": {
            "right": [front_hi_node, back_lo_node],
            "left": [back_hi_node, front_lo_node],
        },
        "handle_node_indices": {
            "right": list(range(front_hi_node, back_lo_node + 1)),
            "left": [*range(back_hi_node, len(path)), front_lo_node],
        },
    }
    # 4 hole locations in the cloth (the flat/arc corners at the rope height)
    holes = {
        "front_left": [-fx, -b, rope_z],
        "front_right": [fx, -b, rope_z],
        "back_right": [fx, b, rope_z],
        "back_left": [-fx, b, rope_z],
    }
    return drawstring, holes


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
def render(verts, faces, layout, out_prefix, p):
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
    ax.add_patch(plt.Circle((-b + sf * 0.5 * t, rz), p["rope_radius"], color=seg_colors["front_tunnel"], zorder=5))
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
    ax.add_patch(plt.Circle((b + sb * 0.5 * t, rz), p["rope_radius"], color=seg_colors["back_tunnel"], zorder=5))
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

    return [
        f"{out_prefix}_oblique.png",
        f"{out_prefix}_top_closeup.png",
        f"{out_prefix}_top.png",
        f"{out_prefix}_hem_section.png",
    ]


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

    obj_path = os.path.join(args.out_dir, "trash_bag.obj")
    json_path = os.path.join(args.out_dir, "trash_bag_layout.json")
    write_obj(obj_path, verts, faces)

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
        "holes": holes,
        "validation": report,
    }
    with open(json_path, "w") as fh:
        json.dump(layout, fh, indent=2)

    print("=" * 70)
    print(f"wrote {obj_path}")
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
    print("=" * 70)

    if args.render:
        pngs = render(verts, faces, layout, os.path.join(args.out_dir, "trash_bag"), p)
        print("rendered:")
        for pp in pngs:
            print("  ", pp)


if __name__ == "__main__":
    main()
