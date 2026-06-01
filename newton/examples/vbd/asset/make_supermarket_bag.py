# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Procedural generator for a supermarket T-shirt / vest carrier bag.

The output is a single, fully welded triangle-mesh thin shell suitable for VBD
cloth simulation, shaped like a classic "thank you" grocery bag:

  * a rectangular body (front / back / left / right walls + bottom) that gives
    the bag depth so contents settle into a real volume;
  * a smooth U-shaped notch scooped into the center of the top edge -- the mouth
    that contents drop through (the mouth boundary follows the scoop curve, so
    it is a clean continuous edge with no stair-stepping);
  * two handles at the left and right ends, formed by the front and back panels
    continuing straight up (in-plane with the body) into straps that pinch
    toward each other and join at the top, so the bag can be carried / pinned by
    the handle tops.

The top edge conforms to the scoop curve by warping each panel column's row
heights to land exactly on the curve (uniform column width, smoothly varying
row height) -- this keeps the boundary smooth while staying crack-free.

Every seam vertex (wall-to-wall, wall-to-bottom, strap-to-body, strap-to-cap) is
welded (shared) through a coordinate-keyed vertex map, so the cloth constraints
are continuous across the whole bag with no detached panels.

Usage::

    python make_supermarket_bag.py                        # writes supermarket_bag.obj
    python make_supermarket_bag.py --cell 0.014           # finer mesh
    python make_supermarket_bag.py --preview preview.png  # also render a PNG
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supermarket_bag.obj")


def _round_key(x, y, z, ndigits=6):
    return (round(float(x), ndigits), round(float(y), ndigits), round(float(z), ndigits))


class _Welder:
    """Accumulates vertices, returning a shared index for coincident positions."""

    def __init__(self):
        self.vmap: dict = {}
        self.verts: list = []

    def add(self, x, y, z):
        key = _round_key(x, y, z)
        idx = self.vmap.get(key)
        if idx is None:
            idx = len(self.verts)
            self.vmap[key] = idx
            self.verts.append((float(x), float(y), float(z)))
        return idx


def _add_quad(welder, faces, p00, p10, p11, p01, outward):
    """Add quad (ring order p00->p10->p11->p01) as two tris oriented toward `outward`."""
    a = np.asarray(p00, dtype=np.float64)
    b = np.asarray(p10, dtype=np.float64)
    d = np.asarray(p01, dtype=np.float64)
    n = np.cross(b - a, d - a)
    if np.dot(n, np.asarray(outward, dtype=np.float64)) < 0.0:
        p10, p01 = p01, p10  # reverse winding so the normal faces `outward`
    i00 = welder.add(*p00)
    i10 = welder.add(*p10)
    i11 = welder.add(*p11)
    i01 = welder.add(*p01)
    faces.append((i00, i10, i11))
    faces.append((i00, i11, i01))


def generate_bag(
    width: float = 0.30,
    depth: float = 0.11,
    body_height: float = 0.30,
    notch_depth: float = 0.07,
    handle_height: float = 0.12,
    handle_width: float = 0.060,
    handle_pinch: float = 0.55,
    cell: float = 0.016,
    z_floor: float = 0.0,
):
    """Build the welded T-shirt-bag mesh.

    Args:
        width: bag size along x [m] (the wide face).
        depth: bag size along y [m] (front-to-back).
        body_height: height of the body walls along z [m]; the handle bases /
            shoulders sit at this height.
        notch_depth: how far the central mouth dips below the shoulders [m].
        handle_height: how far the handle tops rise above the shoulders [m].
        handle_width: width of each handle strap along x [m].
        handle_pinch: fraction by which the front/back straps pinch toward each
            other in y from shoulder to handle top (0 = vertical, 1 = meet).
        cell: target triangle edge length [m]; per-region resolution is derived
            from this so triangles stay roughly uniform.
        z_floor: z of the bag bottom [m].

    Returns:
        (vertices [V,3] float64, faces [F,3] int64, meta dict).
    """
    half_x = 0.5 * width
    half_y = 0.5 * depth
    nx = max(6, round(width / cell))
    ny = max(2, round(depth / cell))
    cx = width / nx
    r_body = max(4, round(body_height / cell))
    r_handle = max(2, round(handle_height / cell))

    z0 = z_floor
    z_shoulder = z0 + body_height
    z_handle_top = z_shoulder + handle_height
    z_mouth = z_shoulder - notch_depth

    X = [-half_x + i * cx for i in range(nx + 1)]
    Y = [-half_y + j * (depth / ny) for j in range(ny + 1)]

    hw_cells = max(2, round(handle_width / cx))
    hw_cells = min(hw_cells, nx // 2 - 1)     # leave a central mouth region
    x_inner = half_x - hw_cells * cx           # x of the handle inner edge
    handle_cols_left = list(range(0, hw_cells))
    handle_cols_right = list(range(nx - hw_cells, nx))

    def scoop(x):
        """Smooth body top edge: flat shoulders at the ends, cosine dip to the mouth."""
        ax = abs(x)
        if ax >= x_inner - 1e-9:
            return z_shoulder
        s = ax / x_inner
        return z_mouth + (z_shoulder - z_mouth) * (0.5 - 0.5 * math.cos(math.pi * s))

    b_top = [scoop(X[i]) for i in range(nx + 1)]

    def body_z(i, r):
        """Warped body z at grid line i, row r: bottom flat, top lands on the curve."""
        return z0 + (b_top[i] - z0) * (r / r_body)

    def side_z(r):
        return z0 + body_height * (r / r_body)

    def strap_z(r):
        return z_shoulder + handle_height * (r / r_handle)

    def pinch_y(base_y, r):
        return base_y * (1.0 - handle_pinch * (r / r_handle))

    w = _Welder()
    faces: list = []

    # ---- bottom (outward -z) ----
    for i in range(nx):
        for j in range(ny):
            _add_quad(
                w, faces,
                (X[i], Y[j], z0),
                (X[i + 1], Y[j], z0),
                (X[i + 1], Y[j + 1], z0),
                (X[i], Y[j + 1], z0),
                outward=(0.0, 0.0, -1.0),
            )

    # ---- front + back walls, warped so the top conforms to the scoop curve ----
    for i in range(nx):
        for r in range(r_body):
            zl0, zl1 = body_z(i, r), body_z(i, r + 1)
            zr0, zr1 = body_z(i + 1, r), body_z(i + 1, r + 1)
            _add_quad(
                w, faces,
                (X[i], -half_y, zl0),
                (X[i + 1], -half_y, zr0),
                (X[i + 1], -half_y, zr1),
                (X[i], -half_y, zl1),
                outward=(0.0, -1.0, 0.0),
            )
            _add_quad(
                w, faces,
                (X[i], half_y, zl0),
                (X[i + 1], half_y, zr0),
                (X[i + 1], half_y, zr1),
                (X[i], half_y, zl1),
                outward=(0.0, 1.0, 0.0),
            )

    # ---- left + right walls (only up to the shoulders) ----
    for j in range(ny):
        for r in range(r_body):
            z_lo, z_hi = side_z(r), side_z(r + 1)
            _add_quad(
                w, faces,
                (-half_x, Y[j], z_lo),
                (-half_x, Y[j + 1], z_lo),
                (-half_x, Y[j + 1], z_hi),
                (-half_x, Y[j], z_hi),
                outward=(-1.0, 0.0, 0.0),
            )
            _add_quad(
                w, faces,
                (half_x, Y[j], z_lo),
                (half_x, Y[j + 1], z_lo),
                (half_x, Y[j + 1], z_hi),
                (half_x, Y[j], z_hi),
                outward=(1.0, 0.0, 0.0),
            )

    # ---- handle straps: front/back panels continue up, pinching toward center ----
    for cols in (handle_cols_left, handle_cols_right):
        for i in cols:
            for r in range(r_handle):
                z_lo, z_hi = strap_z(r), strap_z(r + 1)
                yf0, yf1 = pinch_y(-half_y, r), pinch_y(-half_y, r + 1)
                yb0, yb1 = pinch_y(half_y, r), pinch_y(half_y, r + 1)
                _add_quad(
                    w, faces,
                    (X[i], yf0, z_lo),
                    (X[i + 1], yf0, z_lo),
                    (X[i + 1], yf1, z_hi),
                    (X[i], yf1, z_hi),
                    outward=(0.0, -1.0, 0.0),
                )
                _add_quad(
                    w, faces,
                    (X[i], yb0, z_lo),
                    (X[i + 1], yb0, z_lo),
                    (X[i + 1], yb1, z_hi),
                    (X[i], yb1, z_hi),
                    outward=(0.0, 1.0, 0.0),
                )

    # ---- handle top caps: join the pinched front + back strap tops into a loop ----
    zt = z_handle_top
    yf_top = pinch_y(-half_y, r_handle)
    yb_top = pinch_y(half_y, r_handle)
    for cols in (handle_cols_left, handle_cols_right):
        for i in cols:
            for j in range(ny):
                f0 = (Y[j] + half_y) / depth
                f1 = (Y[j + 1] + half_y) / depth
                ya0 = yf_top + (yb_top - yf_top) * f0
                ya1 = yf_top + (yb_top - yf_top) * f1
                _add_quad(
                    w, faces,
                    (X[i], ya0, zt),
                    (X[i + 1], ya0, zt),
                    (X[i + 1], ya1, zt),
                    (X[i], ya1, zt),
                    outward=(0.0, 0.0, 1.0),
                )

    vertices = np.array(w.verts, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int64)
    meta = {
        "width": width,
        "depth": depth,
        "body_height": body_height,
        "notch_depth": notch_depth,
        "handle_height": handle_height,
        "handle_width": handle_width,
        "handle_pinch": handle_pinch,
        "cell": cell,
        "nx": nx,
        "ny": ny,
        "r_body": r_body,
        "r_handle": r_handle,
        "z_floor": z0,
        "z_shoulder": z_shoulder,
        "z_mouth": z_mouth,
        "z_handle_top": z_handle_top,
        "pin_z_threshold": z_shoulder + 0.85 * handle_height,
        "handle_x_left": [round(X[0], 5), round(X[hw_cells], 5)],
        "handle_x_right": [round(X[nx - hw_cells], 5), round(X[nx], 5)],
    }
    return vertices, faces_arr, meta


def write_obj(path, vertices, faces, meta=None):
    lines = []
    if meta is not None:
        lines.append("# Supermarket T-shirt / vest carrier bag (smooth U-notch mouth, two side handles)")
        lines.append("# Generated by make_supermarket_bag.py")
        for key in (
            "width", "depth", "body_height", "notch_depth", "handle_height",
            "handle_width", "handle_pinch", "cell", "nx", "ny", "r_body", "r_handle",
            "z_floor", "z_shoulder", "z_mouth", "z_handle_top", "pin_z_threshold",
        ):
            lines.append(f"# {key}: {meta[key]}")
        lines.append(f"# handle_x_left: {meta['handle_x_left']}")
        lines.append(f"# handle_x_right: {meta['handle_x_right']}")
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _preview(path, vertices, faces, z_shoulder):
    """Render a few static views of the mesh to a PNG (matplotlib, headless)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = vertices[faces]
    handle = vertices[faces][:, :, 2].max(axis=1) > z_shoulder + 1e-4
    body = ~handle
    views = [("3/4 perspective", 22, -58), ("front", 6, -90), ("side", 6, 0), ("look into open top", 60, -72)]
    fig = plt.figure(figsize=(20, 5.2))
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    span = float((hi - lo).max()) * 0.55
    for n, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), n + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris[body], facecolor="#6fa8dc", edgecolor="#274b6d", linewidths=0.12, alpha=0.85))
        ax.add_collection3d(Poly3DCollection(tris[handle], facecolor="#e69138", edgecolor="#7a4a10", linewidths=0.2, alpha=0.98))
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(f"supermarket_bag.obj  V={len(vertices)}  F={len(faces)}  (handles=orange)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help="output OBJ path")
    parser.add_argument("--width", type=float, default=0.30)
    parser.add_argument("--depth", type=float, default=0.11)
    parser.add_argument("--body-height", type=float, default=0.30)
    parser.add_argument("--notch-depth", type=float, default=0.07)
    parser.add_argument("--handle-height", type=float, default=0.12)
    parser.add_argument("--handle-width", type=float, default=0.060)
    parser.add_argument("--handle-pinch", type=float, default=0.55)
    parser.add_argument("--cell", type=float, default=0.016)
    parser.add_argument("--preview", default=None, help="optional PNG path for a static render")
    args = parser.parse_args()

    vertices, faces, meta = generate_bag(
        width=args.width,
        depth=args.depth,
        body_height=args.body_height,
        notch_depth=args.notch_depth,
        handle_height=args.handle_height,
        handle_width=args.handle_width,
        handle_pinch=args.handle_pinch,
        cell=args.cell,
    )
    write_obj(args.out, vertices, faces, meta)
    print(f"Wrote {args.out}")
    print(f"  vertices: {len(vertices)}  faces: {len(faces)}")
    print(f"  bounds min: {vertices.min(axis=0)}")
    print(f"  bounds max: {vertices.max(axis=0)}")
    print(f"  meta: {meta}")
    if args.preview:
        _preview(args.preview, vertices, faces, meta["z_shoulder"])
        print(f"Wrote preview {args.preview}")


if __name__ == "__main__":
    main()
