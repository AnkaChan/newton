# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Preview and builder dependencies are optional; keep geometry imports light.
# ruff: noqa: PLC0415
"""Original collision benchmark meshes and explicit local asset inventory.

Experimental helpers. All geometry uses metres and Z up. Generated garments
are synthetic shell models, not measured fabric products. Preview geometry is
the rest state, not a simulated drape. Panel generation uses the existing NumPy and SciPy dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.collision_benchmark.asset_paths import (
    STYLE3D_REVISION,
    asset_cache_directory,
    menagerie_asset_path,
    newton_asset_path,
)

__all__ = [
    "CLOTH_NAMES",
    "SOLID_NAMES",
    "AddedAsset",
    "AssetMesh",
    "add_cloth_asset",
    "add_solid_asset",
    "cached_inventory",
    "cloth_mesh",
    "export_cached_previews",
    "export_library",
    "solid_mesh",
    "validate_asset",
]

CLOTH_NAMES = ("patch", "towel", "sheet", "trousers", "dress", "tshirt", "skirt", "tote", "pouch", "gusset_bag")
SOLID_NAMES = ("block", "pad", "sphere", "capsule", "torus", "branched")
_RESOLUTION = {"coarse": 1, "medium": 2, "fine": 3}
_COLORS = {
    "trousers": (0.18, 0.36, 0.69),
    "dress": (0.75, 0.22, 0.37),
    "tshirt": (0.25, 0.65, 0.66),
    "skirt": (0.59, 0.35, 0.70),
    "tote": (0.82, 0.62, 0.28),
    "pouch": (0.28, 0.60, 0.42),
    "gusset_bag": (0.74, 0.47, 0.26),
    "towel": (0.30, 0.62, 0.80),
    "sheet": (0.75, 0.80, 0.88),
}


@dataclass
class AssetMesh:
    """Rest geometry in SI units. Array indices are local to this mesh."""

    name: str
    vertices: np.ndarray
    """Particle positions [m], shape [vertex_count, 3]."""
    triangles: np.ndarray
    """Surface triangle indices, shape [triangle_count, 3]."""
    tetrahedra: np.ndarray | None = None
    """Volume element indices, shape [tet_count, 4], or None for cloth."""
    landmarks: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class AddedAsset:
    """Builder particle range and global landmark indices for one asset."""

    mesh: AssetMesh
    particle_start: int
    particle_end: int
    landmark_indices: dict[str, list[int]]


def _sewn_panel(name, points, openings, depth, level):
    from scipy.spatial import Delaunay

    original_seams = [(i, (i + 1) % len(points)) for i in range(len(points)) if i not in openings]
    outline = np.asarray(points, dtype=float)
    spacing = float(np.max(np.ptp(outline, axis=0))) / (16 + 8 * level)
    boundary, seams = [], set()
    for i, start in enumerate(outline):
        end = outline[(i + 1) % len(outline)]
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
        for j in range(count):
            a = len(boundary)
            boundary.append(start + (end - start) * j / count)
            if i not in openings:
                seams.add((a, a + 1))
    boundary_count = len(boundary)
    seams = {tuple(sorted((a, b % boundary_count))) for a, b in seams}

    def inside(q):
        result = np.zeros(len(q), dtype=bool)
        for i, a in enumerate(outline):
            b = outline[(i + 1) % len(outline)]
            if a[1] == b[1]:
                continue
            crosses = (a[1] > q[:, 1]) != (b[1] > q[:, 1])
            x_at_y = a[0] + (q[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            result ^= crosses & (q[:, 0] < x_at_y)
        return result

    candidates = []
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    for row, z in enumerate(np.arange(lo[1], hi[1] + spacing, spacing * np.sqrt(3) / 2)):
        for x in np.arange(lo[0] + (row % 2) * spacing / 2, hi[0] + spacing, spacing):
            candidates.append((x, z))
    candidates = np.asarray(candidates)
    distance = np.full(len(candidates), np.inf)
    for i, a in enumerate(outline):
        b = outline[(i + 1) % len(outline)]
        d = b - a
        u = np.clip((candidates - a) @ d / (d @ d), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(candidates - a - u[:, None] * d, axis=1))
    p = np.vstack((boundary, candidates[inside(candidates) & (distance > 0.58 * spacing)]))
    t = Delaunay(p).simplices
    oriented = (p[t[:, 1], 0] - p[t[:, 0], 0]) * (p[t[:, 2], 1] - p[t[:, 0], 1]) - (p[t[:, 1], 1] - p[t[:, 0], 1]) * (
        p[t[:, 2], 0] - p[t[:, 0], 0]
    )
    valid = inside(p[t].mean(axis=1)) & (np.abs(oriented) > 1e-10 * spacing * spacing)
    t, oriented = t[valid], oriented[valid]
    flip = oriented < 0
    t[flip] = t[flip][:, ::-1]
    edges = np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    actual_boundary = {tuple(e) for e in unique[counts == 1]}
    expected_boundary = {tuple(sorted((i, (i + 1) % boundary_count))) for i in range(boundary_count)}
    if actual_boundary != expected_boundary:
        raise ValueError(
            f"Panel triangulation did not preserve all boundary segments: {name}; missing={expected_boundary - actual_boundary}; extra={actual_boundary - expected_boundary}"
        )
    seam_ids = {v for e in seams for v in e}
    # A chord between two seam vertices must not become a four-face edge when
    # the back panel is sewn on. Split such chords and give them interior nodes.
    points_refined = p.tolist()
    split = {}
    for tri in t:
        for a, b in zip(tri, np.roll(tri, -1), strict=True):
            key = tuple(sorted((int(a), int(b))))
            if a in seam_ids and b in seam_ids and key not in seams and key not in split:
                split[key] = len(points_refined)
                points_refined.append(((p[a] + p[b]) * 0.5).tolist())
    triangles_refined = []
    for tri in t:
        boundary = []
        for a, b in zip(tri, np.roll(tri, -1), strict=True):
            boundary.append(int(a))
            key = tuple(sorted((int(a), int(b))))
            if key in split:
                boundary.append(split[key])
        if len(boundary) == 3:
            triangles_refined.append(tuple(tri))
        else:
            center = len(points_refined)
            points_refined.append(p[tri].mean(axis=0).tolist())
            triangles_refined.extend((a, boundary[(i + 1) % len(boundary)], center) for i, a in enumerate(boundary))
    p, t = np.asarray(points_refined), np.asarray(triangles_refined, dtype=np.int32)
    distances = np.full(len(p), np.inf)
    for a, b in original_seams:
        start, end = np.asarray(points[a]), np.asarray(points[b])
        direction = end - start
        u = np.clip(((p - start) @ direction) / (direction @ direction), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(p - (start + u[:, None] * direction), axis=1))
    # Lift the two rest panels apart while keeping the stitched curves coincident.
    offset = depth * np.sin(0.5 * np.pi * np.clip(distances / (2 * depth), 0, 1))
    vertices = np.column_stack((p[:, 0], -offset, p[:, 1])).tolist()
    back = []
    for i, (x, z) in enumerate(p):
        if i in seam_ids:
            back.append(i)
        else:
            back.append(len(vertices))
            vertices.append((x, offset[i], z))
    faces = np.vstack((t, np.asarray(back)[t[:, ::-1]]))
    return (
        np.asarray(vertices),
        faces,
        {"seam": sorted(seam_ids)},
        {
            "construction": "two original cut panels, welded seam vertices, unsewn openings",
            "panel_outline_xz_m": points,
            "open_panel_edges": sorted(openings),
            "panel_half_depth_m": depth,
            "panel_mesher": "Delaunay triangular lattice; boundary segments verified; no simulation",
            "panel_target_spacing_m": spacing,
        },
    )


def _quad_grid(vertices, faces, index, positions, flip=False):
    rows = []
    for row in positions:
        out = []
        for p in row:
            key = tuple(np.round(p, 10))
            if key not in index:
                index[key] = len(vertices)
                vertices.append(p)
            out.append(index[key])
        rows.append(out)
    for j in range(len(rows) - 1):
        for i in range(len(rows[j]) - 1):
            a, b, c, d = rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]
            pair = [(a, b, c), (a, c, d)]
            faces.extend([tuple(reversed(t)) for t in pair] if flip else pair)
    return rows


def _bag(name, level):
    width, height, depth = {"tote": (0.30, 0.35, 0.10), "pouch": (0.15, 0.20, 0.018), "gusset_bag": (0.25, 0.30, 0.12)}[
        name
    ]
    nx, ny, nz = 12 * level, max(2, 4 * level), 12 * level
    xs, ys, zs = (
        np.linspace(-width / 2, width / 2, nx + 1),
        np.linspace(-depth / 2, depth / 2, ny + 1),
        np.linspace(0, height, nz + 1),
    )
    vertices, faces, index = [], [], {}
    gusset = 0.025 if name == "gusset_bag" else 0.0

    def xmap(x, y):
        return x * (1 - 2 * gusset / width * (1 - abs(2 * y / depth)))

    for side in (-1, 1):
        _quad_grid(vertices, faces, index, [[(x, side * depth / 2, z) for x in xs] for z in zs], flip=side == 1)
        _quad_grid(
            vertices, faces, index, [[(xmap(side * width / 2, y), y, z) for y in ys] for z in zs], flip=side == -1
        )
    _quad_grid(vertices, faces, index, [[(xmap(x, y), y, 0.0) for x in xs] for y in ys], flip=True)
    handles = []
    if name != "pouch":
        # Each root spans one coarse rim edge. Refinement adds matching nodes.
        strip_width, radius, rise = width / 12, width / 4, 0.14
        for side in (-1, 1):
            positions = []
            for theta in np.linspace(0, np.pi, 20 * level + 1):
                center = np.array((-radius * np.cos(theta), side * depth / 2, height + rise * np.sin(theta)))
                normal = np.array((rise * np.cos(theta), 0, -radius * np.sin(theta)))
                normal /= np.linalg.norm(normal)
                positions.append(
                    [tuple(center + w * normal) for w in np.linspace(-strip_width / 2, strip_width / 2, level + 1)]
                )
            # Put roots on exact existing rim segments (both end edges weld).
            for end, center_x in ((0, -width / 4), (-1, width / 4)):
                root_center_x = center_x + width / 24
                root_x = np.linspace(root_center_x - strip_width / 2, root_center_x + strip_width / 2, level + 1)
                if end == -1:
                    root_x = root_x[::-1]
                positions[end] = [(float(x), side * depth / 2, height) for x in root_x]
            rows = _quad_grid(vertices, faces, index, positions, flip=side == 1)
            handles.extend(rows[len(rows) // 2])
    v = np.asarray(vertices)
    landmarks = {
        "handle_top": handles,
        "rim": np.flatnonzero(np.isclose(v[:, 2], height)).tolist(),
        "bottom": np.flatnonzero(np.isclose(v[:, 2], 0)).tolist(),
    }
    return (
        v,
        np.asarray(faces),
        landmarks,
        {
            "construction": "welded front, back, side and bottom panels; open rim; rim-attached handle strips",
            "bag_body_dimensions_m": [width, depth, height],
            "gusset_indent_m": gusset,
            "body_grid": [nx, ny, nz],
            "handle_count": 0 if name == "pouch" else 2,
        },
    )


def _complete(name, vertices, triangles, *, tetrahedra=None, landmarks=None, metadata=None, resolution="medium"):
    v, t = np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)
    tetrahedra = None if tetrahedra is None else np.asarray(tetrahedra, dtype=np.int32)
    lm = dict(landmarks or {})
    zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())
    lm.setdefault("top", np.flatnonzero(v[:, 2] >= zmax - max((zmax - zmin) * 0.035, 1.0e-5)).tolist())
    lm.setdefault("bottom", np.flatnonzero(v[:, 2] <= zmin + max((zmax - zmin) * 0.035, 1.0e-5)).tolist())
    lm.setdefault("left", [int(np.argmin(v[:, 0]))])
    lm.setdefault("right", [int(np.argmax(v[:, 0]))])
    payload = v.astype("<f4").tobytes() + t.astype("<i4").tobytes()
    if tetrahedra is not None:
        payload += tetrahedra.astype("<i4").tobytes()
    meta = {
        "name": name,
        "family": "cloth" if tetrahedra is None else "solid",
        "units": "m",
        "up_axis": "Z",
        "source": "original procedural benchmark geometry",
        "license": "Apache-2.0",
        "generator_revision": 2,
        "resolution": resolution,
        "dimensions_m": np.ptp(v, axis=0).tolist(),
        "bounds_m": [v.min(axis=0).tolist(), v.max(axis=0).tolist()],
        "vertex_count": len(v),
        "triangle_count": len(t),
        "tet_count": 0 if tetrahedra is None else len(tetrahedra),
        "geometry_sha256": hashlib.sha256(payload).hexdigest(),
        "color_rgb": _COLORS.get(name, (0.43, 0.65, 0.78)),
        "rest_shape": "generated geometry, no settling or simulated draping",
        "landmarks": lm,
        "synthetic_material": {
            "density_kg_m2": 0.25,
            "particle_radius_m": 0.002,
            "tri_ke": 1000.0,
            "tri_ka": 1000.0,
            "tri_kd": 0.01,
            "edge_ke": 0.01,
            "edge_kd": 0.001,
        }
        if tetrahedra is None
        else {
            "density_kg_m3": 100.0,
            "particle_radius_m": 0.001,
            "k_mu_pa": 20000.0,
            "k_lambda_pa": 20000.0,
            "k_damp_pa_s": 0.01,
        },
    }
    meta.update(metadata or {})
    return AssetMesh(name, v, t, tetrahedra, lm, meta)


def cloth_mesh(name: str, *, resolution: str = "medium") -> AssetMesh:
    """Create original shell geometry [m], with Z up and its bottom at Z=0.

    Garment seams share actual particle indices. Waist, cuff, neck, armhole,
    hem and bag-mouth openings remain open. Geometry resolution does not alter
    physical shell thickness; the builder contact radius is a separate setting.
    """
    level = _RESOLUTION[resolution]
    if name in ("patch", "towel", "sheet"):
        width, height = {"patch": (0.1, 0.1), "towel": (0.4, 0.6), "sheet": (1.0, 1.0)}[name]
        nx, nz = (8 * level, 8 * level) if name == "patch" else (12 * level, 16 * level)
        v, t = [], []
        _quad_grid(
            v,
            t,
            {},
            [[(x, 0.0, z) for x in np.linspace(-width / 2, width / 2, nx + 1)] for z in np.linspace(0, height, nz + 1)],
        )
        landmarks, metadata = {}, {"construction": "single rectangular cloth panel", "grid": [nx, nz]}
    elif name in ("tote", "pouch", "gusset_bag"):
        v, t, landmarks, metadata = _bag(name, level)
    elif name == "trousers":
        points = [
            (-0.18, 0.85),
            (-0.20, 0.48),
            (-0.18, 0),
            (-0.035, 0),
            (-0.025, 0.40),
            (0, 0.46),
            (0.025, 0.40),
            (0.035, 0),
            (0.18, 0),
            (0.20, 0.48),
            (0.18, 0.85),
        ]
        v, t, landmarks, metadata = _sewn_panel(name, points, {2, 7, 10}, 0.025, level)
    elif name == "dress":
        points = [
            (-0.38, 0),
            (0.38, 0),
            (0.18, 0.62),
            (0.15, 0.78),
            (0.18, 0.94),
            (0.105, 1.0),
            (0.065, 0.88),
            (-0.065, 0.88),
            (-0.105, 1.0),
            (-0.18, 0.94),
            (-0.15, 0.78),
            (-0.18, 0.62),
        ]
        v, t, landmarks, metadata = _sewn_panel(name, points, {0, 2, 3, 5, 6, 7, 9, 10}, 0.035, level)
    elif name == "tshirt":
        points = [
            (-0.24, 0),
            (0.24, 0),
            (0.24, 0.40),
            (0.41, 0.31),
            (0.49, 0.49),
            (0.25, 0.65),
            (0.10, 0.68),
            (0.07, 0.61),
            (-0.07, 0.61),
            (-0.10, 0.68),
            (-0.25, 0.65),
            (-0.49, 0.49),
            (-0.41, 0.31),
            (-0.24, 0.40),
        ]
        v, t, landmarks, metadata = _sewn_panel(name, points, {0, 3, 6, 7, 8, 11}, 0.035, level)
    elif name == "skirt":
        points = [(-0.34, 0), (0.34, 0), (0.17, 0.55), (-0.17, 0.55)]
        v, t, landmarks, metadata = _sewn_panel(name, points, {0, 2}, 0.045, level)
    else:
        raise ValueError(f"Unknown cloth asset {name!r}; choose from {CLOTH_NAMES}")
    return _complete(name, v, t, landmarks=landmarks, metadata=metadata, resolution=resolution)


def _surface(tets):
    face_map = {}
    for a, b, c, d in tets:
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(face))
            if key in face_map:
                face_map[key] = None
            else:
                face_map[key] = face
    return np.asarray([f for f in face_map.values() if f is not None], dtype=np.int32)


def _hex_tets(corners):
    a, b, c, d, e, f, g, h = corners
    return [(a, b, c, g), (a, c, d, g), (a, d, h, g), (a, h, e, g), (a, e, f, g), (a, f, b, g)]


def _tet_grid(axes, transform=None, keep=None, periodic=False):
    points = np.asarray([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]])
    shape = tuple(len(a) for a in axes)
    ids = np.arange(len(points)).reshape(shape)
    tets = []
    for i in range(shape[0] if periodic else shape[0] - 1):
        ii = (i + 1) % shape[0]
        for j in range(shape[1] - 1):
            for k in range(shape[2] - 1):
                corners = [
                    ids[i, j, k],
                    ids[ii, j, k],
                    ids[ii, j + 1, k],
                    ids[i, j + 1, k],
                    ids[i, j, k + 1],
                    ids[ii, j, k + 1],
                    ids[ii, j + 1, k + 1],
                    ids[i, j + 1, k + 1],
                ]
                if keep is None or keep(points[corners].mean(axis=0)):
                    tets.extend(_hex_tets(corners))
    tets = np.asarray(tets, dtype=np.int32)
    used, inverse = np.unique(tets, return_inverse=True)
    points, tets = points[used], inverse.reshape(-1, 4)
    if transform is not None:
        points = transform(points)
    determinants = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]])
    inverted = determinants < 0
    tets[inverted, 2], tets[inverted, 3] = tets[inverted, 3].copy(), tets[inverted, 2].copy()
    points[:, 2] -= points[:, 2].min()
    return points, tets


def solid_mesh(name: str, *, resolution: str = "medium") -> AssetMesh:
    """Create an original conforming tet mesh in metres, with bottom Z=0."""
    level = _RESOLUTION[resolution]
    if name in ("block", "pad"):
        dims, cells = {"block": ((0.06, 0.04, 0.03), (4, 3, 2)), "pad": ((0.12, 0.08, 0.01), (8, 6, 1))}[name]
        axes = [np.linspace(-d / 2, d / 2, n * level + 1) for d, n in zip(dims, cells, strict=True)]
        v, tet = _tet_grid(axes)
        construction = "structured hexahedral grid split into six conforming tetrahedra per cell"
    elif name in ("sphere", "capsule"):
        n = 4 * level
        axes = [np.linspace(-1, 1, n + 1)] * 3

        def transform(p):
            length = np.linalg.norm(p, axis=1)
            radius = 0.025 if name == "sphere" else 0.0175
            scale = np.divide(np.max(np.abs(p), axis=1), length, out=np.zeros_like(length), where=length > 0)
            q = radius * p * scale[:, None]
            if name == "capsule":
                q[:, 2] += 0.0225 * p[:, 2]
            return q

        v, tet = _tet_grid(axes, transform)
        construction = "cube grid radially mapped to sphere" + (
            " and axially stretched to a rounded capsule surrogate" if name == "capsule" else ""
        )
    elif name == "torus":
        axes = [
            np.linspace(0, 2 * np.pi, 16 * level, endpoint=False),
            np.linspace(-1, 1, 2 * level + 1),
            np.linspace(-1, 1, 2 * level + 1),
        ]

        def transform(p):
            uv = p[:, 1:]
            r = np.linalg.norm(uv, axis=1)
            s = np.divide(np.max(np.abs(uv), axis=1), r, out=np.zeros_like(r), where=r > 0)
            uv = 0.01 * uv * s[:, None]
            return np.column_stack(((0.03 + uv[:, 0]) * np.cos(p[:, 0]), (0.03 + uv[:, 0]) * np.sin(p[:, 0]), uv[:, 1]))

        v, tet = _tet_grid(axes, transform, periodic=True)
        construction = "periodic swept disk grid with major radius 0.03 m and tube radius 0.01 m"
    elif name == "branched":
        axes = [
            np.linspace(-0.06, 0.06, 12 * level + 1),
            np.linspace(-0.06, 0.06, 12 * level + 1),
            np.linspace(-0.012, 0.012, 2 * level + 1),
        ]
        v, tet = _tet_grid(axes, keep=lambda p: abs(p[0]) < 0.02 or abs(p[1]) < 0.02)
        construction = "original four-arm cross, rectangular arms; conforming union of voxel cells"
    else:
        raise ValueError(f"Unknown solid asset {name!r}; choose from {SOLID_NAMES}")
    return _complete(
        name, v, _surface(tet), tetrahedra=tet, resolution=resolution, metadata={"construction": construction}
    )


def add_cloth_asset(
    builder,
    name: str,
    *,
    resolution: str = "medium",
    pos=(0.0, 0.0, 0.0),
    rot=None,
    scale: float = 1.0,
    density: float = 0.25,
    particle_radius: float = 0.002,
    **material,
) -> AddedAsset:
    """Add an original shell; density is kg/m², radius and position are m.

    Extra keyword arguments go to ModelBuilder.add_cloth_mesh. They may set
    elastic constants, color and velocity. This helper never pins particles.
    """
    import warp as wp

    mesh = cloth_mesh(name, resolution=resolution)
    start = len(builder.particle_q)
    kwargs = {
        "tri_ke": 1000.0,
        "tri_ka": 1000.0,
        "tri_kd": 0.01,
        "edge_ke": 0.01,
        "edge_kd": 0.001,
        "vel": (0.0, 0.0, 0.0),
        "color": mesh.metadata["color_rgb"],
    }
    kwargs.update(material)
    builder.add_cloth_mesh(
        pos=pos,
        rot=wp.quat_identity() if rot is None else rot,
        scale=scale,
        vertices=mesh.vertices.tolist(),
        indices=mesh.triangles.ravel().tolist(),
        density=density,
        particle_radius=particle_radius,
        **kwargs,
    )
    return AddedAsset(
        mesh, start, len(builder.particle_q), {k: [start + i for i in ids] for k, ids in mesh.landmarks.items()}
    )


def add_solid_asset(
    builder,
    name: str,
    *,
    resolution: str = "medium",
    pos=(0.0, 0.0, 0.0),
    rot=None,
    scale: float = 1.0,
    density: float = 100.0,
    particle_radius: float = 0.001,
    **material,
) -> AddedAsset:
    """Add original tet geometry; density is kg/m³, radius and position are m."""
    import warp as wp

    mesh = solid_mesh(name, resolution=resolution)
    start = len(builder.particle_q)
    kwargs = {
        "k_mu": 20000.0,
        "k_lambda": 20000.0,
        "k_damp": 0.01,
        "vel": (0.0, 0.0, 0.0),
        "color": mesh.metadata["color_rgb"],
    }
    kwargs.update(material)
    builder.add_soft_mesh(
        pos=pos,
        rot=wp.quat_identity() if rot is None else rot,
        scale=scale,
        vertices=mesh.vertices.tolist(),
        indices=mesh.tetrahedra.ravel().tolist(),
        density=density,
        particle_radius=particle_radius,
        **kwargs,
    )
    return AddedAsset(
        mesh, start, len(builder.particle_q), {k: [start + i for i in ids] for k, ids in mesh.landmarks.items()}
    )


def validate_asset(mesh: AssetMesh) -> dict:
    """Check finite coordinates, connectivity, manifold edges and tet volume."""
    v, t = mesh.vertices.astype(float), mesh.triangles
    area = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    lengths = np.stack([np.linalg.norm(v[t[:, (i + 1) % 3]] - v[t[:, i]], axis=1) for i in range(3)], axis=1)
    angles = []
    for i in range(3):
        a, b, c = lengths[:, i], lengths[:, (i + 1) % 3], lengths[:, (i + 2) % 3]
        cosine = np.divide(a * a + b * b - c * c, 2 * a * b, out=np.ones_like(a), where=a * b > 0)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    minimum_angles = np.min(angles, axis=0)
    edges = np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    directed_edges = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1))
    parent = np.arange(len(v))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    elements = t if mesh.tetrahedra is None else mesh.tetrahedra
    for element in elements:
        for j in element[1:]:
            parent[find(j)] = find(element[0])
    components = len({find(i) for i in range(len(v))})
    result = {
        "finite": bool(np.isfinite(v).all()),
        "min_triangle_area_m2": float(area.min()),
        "surface_area_m2": float(area.sum()),
        "min_triangle_angle_degrees": float(minimum_angles.min()),
        "triangles_with_angle_below_10_degrees": int(np.sum(minimum_angles < 10)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "boundary_edges": int(np.sum(counts == 1)),
        "inconsistent_winding_edges": int(np.sum((counts == 2) & (orientation != 0))),
        "connected_components": components,
    }
    if mesh.tetrahedra is not None:
        tet = mesh.tetrahedra
        volumes = np.linalg.det(v[tet[:, 1:]] - v[tet[:, :1]]) / 6
        result.update(
            min_tet_volume_m3=float(volumes.min()),
            volume_m3=float(volumes.sum()),
            nonpositive_tets=int(np.sum(volumes <= 0)),
        )
    result["valid"] = bool(
        result["finite"]
        and area.min() > 1e-14
        and result["nonmanifold_edges"] == 0
        and result["inconsistent_winding_edges"] == 0
        and components == 1
        and result.get("nonpositive_tets", 0) == 0
    )
    return result


def _file_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def _source_record(checkout, directory):
    head = (checkout / ".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = checkout / ".git" / head[5:]
        head = ref.read_text().strip() if ref.exists() else head
    digest = hashlib.sha256()
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(directory))
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            digest.update((relative + "\0" + file_hash + "\n").encode())
            files.append({"path": relative, "sha256": file_hash})
    return {"source_revision": head, "source_tree_sha256": digest.hexdigest(), "source_files": files}


def cached_inventory(cache_dir: str | Path | None = None) -> list[dict]:
    """Inventory local sources without downloading or implying solver support."""
    cache = asset_cache_directory(cache_dir)
    specs = [
        (
            "franka_fr3_hand",
            "newton-assets_franka_emika_panda_*_f8fb7abc",
            "franka_emika_panda",
            "urdf/fr3_franka_hand.urdf",
        ),
        ("robotiq_2f85", "mujoco_menagerie_robotiq_2f85_v4_*_da76818e", "robotiq_2f85_v4", "2f85.xml"),
        ("allegro", "mujoco_menagerie_wonik_allegro_*_da76818e", "wonik_allegro", "right_hand.xml"),
        ("leap", "mujoco_menagerie_leap_hand_*_da76818e", "leap_hand", "right_hand.xml"),
        ("shadow", "mujoco_menagerie_shadow_hand_*_da76818e", "shadow_hand", "right_hand.xml"),
        (
            "rubber_duck",
            "newton-assets_manipulation_objects_rubber_duck_*_f8fb7abc",
            "manipulation_objects/rubber_duck",
            "model.usda",
        ),
    ]
    records = []
    for name, pattern, folder, main in specs:
        found = sorted(cache.glob(pattern))
        if not found:
            records.append({"name": name, "status": "missing"})
            continue
        directory = found[-1] / folder
        entry = directory / main
        license_path = directory / "LICENSE"
        record = {
            "name": name,
            "status": "cached, import and grasp qualification separate",
            "directory": str(directory),
            "cache_revision_suffix": found[-1].name.rsplit("_", 1)[-1],
            "units": "m (robot convention; verify importer)",
            "source": "https://github.com/google-deepmind/mujoco_menagerie"
            if "menagerie" in pattern
            else "https://github.com/newton-physics/newton-assets",
        }
        record.update(_source_record(found[-1], directory))
        record["license"] = {
            "franka_fr3_hand": "Apache-2.0",
            "robotiq_2f85": "BSD-2-Clause",
            "allegro": "BSD-2-Clause",
            "leap": "MIT",
            "shadow": "Apache-2.0",
            "rubber_duck": "Apache-2.0",
        }[name]
        record["entrypoint"] = _file_record(entry) if entry.exists() else {"requested": main, "status": "not found"}
        if license_path.exists():
            record["license_file"] = _file_record(license_path)
            record["license_notice"] = license_path.read_text()
        records.append(record)
    found = sorted(cache.glob(f"newton-assets_style3d_*_{STYLE3D_REVISION[:8]}"))
    if found:
        directory = found[-1] / "style3d"
        for name in ("Female_T_Shirt", "Women_Sweatshirt", "Women_Skirt", "h1_jacket", "h1_cake_skirt"):
            entry = directory / "garments" / f"{name}.usd"
            if entry.exists():
                records.append(_style3d_record(directory, name))
    return records


def _style3d_record(directory, name):
    record = {
        "name": name,
        "status": "cached source; Style3D seam/material mapping not transferred",
        "source": "https://github.com/newton-physics/newton-assets",
        "cache_revision_suffix": STYLE3D_REVISION[:8],
        "entrypoint": _file_record(directory / "garments" / f"{name}.usd"),
        "license": "BSD-3-Clause",
        "license_file": _file_record(directory / "LICENSE"),
        "license_notice": (directory / "LICENSE").read_text(),
        "units": "USD stage metresPerUnit=1, Y up; pattern UV coordinates are mm per source README",
    }
    record.update(_source_record(directory.parent, directory))
    return record


def _preview(mesh, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(5, 5), facecolor="#f4f5f7")
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("#f4f5f7")
    vertices = mesh.vertices
    collection = Poly3DCollection(
        vertices[mesh.triangles],
        facecolor=mesh.metadata["color_rgb"],
        edgecolor=(0.12, 0.18, 0.24, 0.25) if len(mesh.triangles) < 10000 else "none",
        linewidth=0.18,
        alpha=1.0,
    )
    ax.add_collection3d(collection)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    size = float(np.max(hi - lo)) * 0.56
    ax.set(
        xlim=(center[0] - size, center[0] + size),
        ylim=(center[1] - size, center[1] + size),
        zlim=(center[2] - size, center[2] + size),
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=17, azim=-62)
    pose = mesh.metadata.get("preview_pose", "rest geometry")
    ax.set_title(f"{mesh.name.replace('_', ' ').title()}\n{len(vertices):,} vertices · {pose}", fontsize=11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _cached_mesh(record):
    import warp as wp

    import newton

    path = record["entrypoint"]["path"]
    vertex_parts, triangle_parts = [], []
    count = 0
    if path.endswith((".usd", ".usda")):
        from pxr import Usd, UsdGeom

        import newton.usd

        stage = Usd.Stage.Open(path)
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or not UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get():
                continue
            mesh = newton.usd.get_mesh(prim, compute_inertia=False, load_visual_materials=False)
            matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            vertices = np.column_stack((mesh.vertices, np.ones(len(mesh.vertices)))) @ matrix
            vertices = vertices[:, :3]
            if UsdGeom.GetStageUpAxis(stage) == "Y":
                vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
            vertex_parts.append(vertices)
            triangle_parts.append(np.asarray(mesh.indices).reshape(-1, 3) + count)
            count += len(vertices)
        extra = {
            "preview_pose": "authored static geometry",
            "units": "m",
            "up_axis": "Z",
            "usd_stage_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "usd_authored_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        }
    else:
        import trimesh

        builder = newton.ModelBuilder()
        with wp.ScopedDevice("cpu"):
            if path.endswith(".urdf"):
                builder.add_urdf(path, floating=False)
            else:
                builder.add_mjcf(path, floating=False, parse_sites=False)
            model = builder.finalize(device="cpu")
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        for i, kind in enumerate(builder.shape_type):
            if not builder.shape_flags[i] & newton.ShapeFlags.VISIBLE:
                continue
            scale = np.asarray(builder.shape_scale[i])
            if kind in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
                src = builder.shape_source[i]
                v = np.asarray(src.vertices) * scale
                t = np.asarray(src.indices).reshape(-1, 3)
            else:
                if kind == newton.GeoType.BOX:
                    src = trimesh.creation.box(extents=2 * scale)
                elif kind == newton.GeoType.SPHERE:
                    src = trimesh.creation.icosphere(subdivisions=2, radius=scale[0])
                elif kind == newton.GeoType.CAPSULE:
                    src = trimesh.creation.capsule(radius=scale[0], height=2 * scale[1], count=(12, 12))
                elif kind == newton.GeoType.CYLINDER:
                    src = trimesh.creation.cylinder(radius=scale[0], height=2 * scale[1], sections=16)
                else:
                    continue
                v, t = np.asarray(src.vertices), np.asarray(src.faces)
            local = wp.transform(*builder.shape_transform[i])
            body = builder.shape_body[i]
            xform = local if body < 0 else wp.transform_multiply(wp.transform(*body_q[body]), local)
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(xform))).reshape(3, 3)
            vertices = v @ rotation.T + np.asarray(wp.transform_get_translation(xform))
            vertex_parts.append(vertices)
            triangle_parts.append(t + count)
            count += len(vertices)
        extra = {
            "preview_pose": "imported joint rest pose",
            "units": "m",
            "up_axis": "Z",
            "body_count": model.body_count,
            "joint_count": model.joint_count,
            "shape_count": model.shape_count,
            "preview_geometry": "visible imported shapes; forward kinematics only, no simulation",
        }
    if not vertex_parts:
        raise ValueError(f"No preview mesh in {path}")
    v, t = np.vstack(vertex_parts), np.vstack(triangle_parts)
    v[:, 2] -= v[:, 2].min()
    mesh = _complete(
        record["name"],
        v,
        t,
        metadata={
            "source": record["source"],
            "license": record["license"],
            "synthetic_material": None,
            "color_rgb": (0.52, 0.59, 0.66) if not path.endswith((".usd", ".usda")) else (0.76, 0.47, 0.30),
            **extra,
        },
    )
    return mesh


def export_cached_previews(output: str | Path) -> list[dict]:
    """Fetch pinned optional sources and render their geometry without simulation.

    Uses optional local Newton/USD/trimesh/matplotlib packages. Source hashes and
    notices accompany the previews. Garment solver/material mappings remain
    unqualified; a static preview does not imply contact support.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("franka_emika_panda", "manipulation_objects/rubber_duck"):
        newton_asset_path(folder)
    newton_asset_path("style3d", revision=STYLE3D_REVISION)
    for folder in ("robotiq_2f85_v4", "wonik_allegro", "leap_hand", "shadow_hand"):
        menagerie_asset_path(folder)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = cached_inventory()
    for record in records:
        if "path" not in record.get("entrypoint", {}):
            continue
        try:
            mesh = _cached_mesh(record)
            filename = f"cached_{record['name']}.png"
            _preview(mesh, output / filename)
            record.update(
                preview=filename,
                dimensions_m=mesh.metadata["dimensions_m"],
                bounds_m=mesh.metadata["bounds_m"],
                vertex_count=len(mesh.vertices),
                triangle_count=len(mesh.triangles),
                preview_geometry_sha256=mesh.metadata["geometry_sha256"],
                preview_pose=mesh.metadata["preview_pose"],
                preview_disclosure="static imported geometry; no simulation or material qualification",
            )
            for key in (
                "body_count",
                "joint_count",
                "shape_count",
                "usd_stage_meters_per_unit",
                "usd_authored_up_axis",
            ):
                if key in mesh.metadata:
                    record[key] = mesh.metadata[key]
            print(f"Cached preview: {record['name']}, {len(mesh.vertices)} vertices", flush=True)
        except Exception as exc:
            record["preview_error"] = f"{type(exc).__name__}: {exc}"
            print(f"Cached preview unavailable: {record['name']}: {exc}", flush=True)
    manifest["cached_assets"] = records
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _write_gallery(output, manifest)
    return records


def _write_gallery(output, manifest):
    import html

    def card(record, cached=False):
        name = record["name"]
        metadata_file = f"{'cached_' if cached else ''}{name}.metadata.json"
        (output / metadata_file).write_text(json.dumps(record, indent=2) + "\n")
        title = name.replace("_", " ").title()
        dimensions = " &times; ".join(f"{x:.3f}" for x in record.get("dimensions_m", []))
        image = (
            f'<img src="{record["preview"]}" alt="{html.escape(title)} static geometry" loading="lazy">'
            if "preview" in record
            else ""
        )
        construction = record.get("construction", record.get("preview_pose", record.get("status", "")))
        source = record["source"]
        link = f'<a href="{html.escape(source)}">Source</a>' if source.startswith("https://") else html.escape(source)
        detail = f"{record.get('vertex_count', 0):,} vertices · {record.get('triangle_count', 0):,} triangles"
        if record.get("tet_count"):
            detail += f" · {record['tet_count']:,} tetrahedra"
        downloads = (
            "" if cached else f' · <a href="{record["npz"]}">NPZ</a> · <a href="{record["obj"]}">OBJ surface</a>'
        )
        return (
            f"<figure>{image}<figcaption><h3>{html.escape(title)}</h3><p>{dimensions} m (x &times; y &times; z)<br>{detail}</p>"
            f"<p>{html.escape(construction)}.</p><p>{link} · {record.get('license', 'See notice')}</p>"
            f'<p><a href="{metadata_file}">Metadata and hash</a>{downloads}</p></figcaption></figure>'
        )

    generated = "".join(card(r) for r in manifest.get("generated_assets", []))
    cached = "".join(card(r, True) for r in manifest.get("cached_assets", []))
    derived = "".join(card(r, True) for r in manifest.get("derived_assets", []))
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Collision benchmark assets</title>'
        "<style>body{font:16px/1.5 system-ui;margin:clamp(1rem,3vw,3rem);background:#f4f5f7;color:#172635;max-width:1800px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:1rem}figure{margin:0;background:white;padding:1rem;border-radius:12px;overflow:hidden}"
        "img{width:100%;height:auto}h1{font-size:28px}h2{font-size:24px;margin-top:2rem}h3{margin:0}a{color:#165b92}p{margin:.6rem 0}.note{max-width:1000px}</style></head><body>"
        '<h1>Collision benchmark assets</h1><p class="note">Static rest geometry and source inventory. These images are not physics results. '
        "All displayed dimensions use metres. Smoke scenes can apply a different declared scale, pose, and material. "
        'Use each scene manifest for its simulated dimensions.</p><p><a href="manifest.json">Full manifest and license notices</a> · '
        '<a href="README.md">Helper API and limits</a></p><h2>Original generated meshes</h2><p class="note">'
        "Garments use welded panel seams with open cuffs, hems, necks, and armholes. Bags have sewn bottoms, open rims, and attached handles. "
        'Material presets are synthetic. They do not represent measured cotton, denim, or rubber.</p><div class="grid">'
        + generated
        + "</div>"
        '<h2>Cached source geometry</h2><p class="note">Robot images use imported geometry and forward kinematics at the source rest pose. '
        "Garment images show authored source meshes. A preview does not establish a working grasp, a controller, or a Style3D-to-VBD material mapping. "
        'The Franka source contains the FR3 arm and hand.</p><div class="grid">' + cached + "</div>"
        '<h2>Prepared source derivatives</h2><p class="note">These meshes are separate from original geometry. '
        'Metadata records preprocessing, topology checks, source notices, and remaining quality limits.</p><div class="grid">'
        + derived
        + "</div>"
        '<h2>Unresolved candidates</h2><p class="note">The local LIDO / TetWild / Thingi10K subset, bunny, and octopus have not been copied. '
        "Individual source identities or reuse terms remain unresolved. Original primitives and the four-arm cross provide traceable coverage.</p></body></html>"
    )


def export_library(output: str | Path, *, resolution: str = "medium", previews: bool = True) -> dict:
    """Write generated NPZ/OBJ assets, source manifest and static PNG previews."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous_path = output / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    records = []
    for name in CLOTH_NAMES + SOLID_NAMES:
        mesh = (
            cloth_mesh(name, resolution=resolution) if name in CLOTH_NAMES else solid_mesh(name, resolution=resolution)
        )
        checks = validate_asset(mesh)
        if not checks["valid"]:
            raise ValueError(f"Invalid generated asset {name}: {checks}")
        arrays = {"vertices": mesh.vertices, "triangles": mesh.triangles}
        if mesh.tetrahedra is not None:
            arrays["tetrahedra"] = mesh.tetrahedra
        np.savez_compressed(output / f"{name}.npz", **arrays)
        with (output / f"{name}.obj").open("w") as f:
            f.write("# Original Newton collision benchmark geometry; Apache-2.0; units m\n")
            for v in mesh.vertices:
                f.write("v " + " ".join(f"{x:.9g}" for x in v) + "\n")
            for tri in mesh.triangles + 1:
                f.write("f " + " ".join(str(x) for x in tri) + "\n")
        record = dict(mesh.metadata, validation=checks, npz=f"{name}.npz", obj=f"{name}.obj")
        if previews:
            _preview(mesh, output / f"{name}.png")
            record["preview"] = f"{name}.png"
        records.append(record)
    cached = previous.get("cached_assets", [])
    if not any("preview" in record for record in cached):
        cached = cached_inventory()
    manifest = {
        "schema_version": 1,
        "generated_assets": records,
        "cached_assets": cached,
        "derived_assets": previous.get("derived_assets", []),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preview_disclosure": "All PNGs show generated rest geometry. They are not simulation results.",
        "excluded_candidates": [
            {
                "name": "LIDO TetWild / Thingi10K six-object subset",
                "reason": "Individual source model rights and semantic identities are unresolved; not copied or relabeled.",
            },
            {
                "name": "local bunny.vtk and octopus.vtk",
                "reason": "Source provenance and redistribution terms not established.",
            },
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_gallery(output, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("assets"))
    parser.add_argument("--resolution", choices=tuple(_RESOLUTION), default="medium")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--cached-previews", action="store_true")
    args = parser.parse_args()
    result = export_library(args.output, resolution=args.resolution, previews=not args.no_previews)
    if args.cached_previews:
        result["cached_assets"] = export_cached_previews(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "generated_assets": len(result["generated_assets"]),
                "cached_assets": len(result["cached_assets"]),
            }
        )
    )
