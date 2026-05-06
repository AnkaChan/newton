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

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence

import numpy as np

import newton.examples

DEFAULT_PROXY_MODE = "cgal-isotropic-remesh"
PROXY_MODES = (
    DEFAULT_PROXY_MODE,
    "meshlab-isotropic-remesh",
    "surface-decimate",
    "qem-decimate",
)
_LOG_PREFIX = "[bag_mesh]"


def build_bary_map(
    full_verts: np.ndarray,
    phys_verts: np.ndarray,
    phys_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map each full-resolution vertex to the closest physics triangle."""
    from scipy.spatial import cKDTree

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

    _, nearest = tree.query(full_verts, k=min(5, len(centroids)))
    if nearest.ndim == 1:
        nearest = nearest[:, None]

    for i in range(n_full):
        p = full_verts[i]
        best_dist = 1e30
        best_bary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        best_tri = 0
        for tri_idx in nearest[i]:
            a, b, c = v0[tri_idx], v1[tri_idx], v2[tri_idx]
            edge0 = b - a
            edge1 = c - a
            delta = p - a
            d00 = edge0 @ edge0
            d01 = edge0 @ edge1
            d11 = edge1 @ edge1
            dv0 = delta @ edge0
            dv1 = delta @ edge1
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                continue
            u = (d11 * dv0 - d01 * dv1) / denom
            w = (d00 * dv1 - d01 * dv0) / denom
            t = 1.0 - u - w
            t = max(0.0, min(1.0, t))
            u = max(0.0, min(1.0, u))
            w = max(0.0, min(1.0, w))
            total = t + u + w
            if total > 0.0:
                t /= total
                u /= total
                w /= total
            proj = a * t + b * u + c * w
            dist = float(np.sum((p - proj) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_bary = np.array([t, u, w], dtype=np.float32)
                best_tri = int(tri_idx)
        vi0[i] = phys_faces[best_tri, 0]
        vi1[i] = phys_faces[best_tri, 1]
        vi2[i] = phys_faces[best_tri, 2]
        bary[i] = best_bary

    return vi0, vi1, vi2, bary


def build_bary_map_with_logging(
    full_verts: np.ndarray,
    phys_verts: np.ndarray,
    phys_faces: np.ndarray,
    *,
    log_prefix: str = _LOG_PREFIX,
    message: str = "Building barycentric render map",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a barycentric render-to-simulation map with uniform progress output."""
    prefix = _log_prefix(log_prefix)
    print(f"{prefix}{message}...", end=" ", flush=True)
    bary_map = build_bary_map(full_verts, phys_verts, phys_faces)
    print("done.")
    return bary_map


def log_mesh_counts(
    *,
    log_prefix: str = _LOG_PREFIX,
    full_label: str = "Full mesh",
    full_verts: np.ndarray,
    full_faces: np.ndarray,
    face_name: str = "tris",
) -> None:
    """Print uniform full/source mesh counts."""
    prefix = _log_prefix(log_prefix)
    print(
        f"{prefix}{full_label}: {len(full_verts)} verts, "
        f"{len(full_faces)} {face_name}"
    )


def load_kfc_mesh_zup(bag_height_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """Load the KFC bag mesh, rotate it to Z-up, and scale it to cm."""
    from pxr import Usd, UsdGeom

    usd_path = str(newton.examples.get_asset("kfc.usd"))
    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetPrimAtPath("/World/material/material_001")
    usd_mesh = UsdGeom.Mesh(prim)

    points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
    faces = np.array(
        usd_mesh.GetFaceVertexIndicesAttr().Get(),
        dtype=np.int32,
    ).reshape(-1, 3)

    points_zup = np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])
    usd_height_m = float(points_zup[:, 2].max() - points_zup[:, 2].min())
    scale_m = (bag_height_cm / 100.0) / usd_height_m
    points_zup *= scale_m
    points_zup[:, 2] -= float(points_zup[:, 2].min())

    return (points_zup * 100.0).astype(np.float32), faces


def add_proxy_mesh_arguments(parser) -> None:
    """Add shared bag proxy mesh CLI flags."""
    parser.add_argument(
        "--target-faces",
        type=int,
        default=1200,
        help="Target proxy face count for solver-side bag simulation",
    )
    parser.add_argument(
        "--proxy-mode",
        type=str,
        default=DEFAULT_PROXY_MODE,
        choices=PROXY_MODES,
        help="Proxy mesh generation method",
    )


def decimate_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    proxy_mode: str = DEFAULT_PROXY_MODE,
    *,
    make_intersection_free: bool = False,
    intersection_checker: Callable[[np.ndarray, np.ndarray], Sequence[tuple[int, int]]] | None = None,
    checker_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    edge_edge_min_distance: float = 0.0,
    intersection_free_min_area: float = 1.0e-12,
    log_prefix: str = _LOG_PREFIX,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a lower-resolution bag proxy mesh for solver-side simulation."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    target_faces = max(int(target_faces), 1)

    if proxy_mode == DEFAULT_PROXY_MODE:
        proxy_verts, proxy_faces = _cgal_isotropic_remesh(verts, faces, target_faces)
    elif proxy_mode == "meshlab-isotropic-remesh":
        proxy_verts, proxy_faces = _meshlab_isotropic_remesh(verts, faces, target_faces)
    elif proxy_mode == "surface-decimate":
        proxy_verts, proxy_faces = _surface_decimate(verts, faces, target_faces)
    elif proxy_mode == "qem-decimate":
        proxy_verts, proxy_faces = _qem_decimate(verts, faces, target_faces)
    else:
        raise ValueError(f"Unknown proxy mode: {proxy_mode}")

    if make_intersection_free:
        proxy_verts, proxy_faces = make_proxy_mesh_intersection_free(
            proxy_verts,
            proxy_faces,
            intersection_checker=intersection_checker,
            checker_transform=checker_transform,
            edge_edge_min_distance=edge_edge_min_distance,
            min_area=intersection_free_min_area,
            log_prefix=log_prefix,
        )
        _log_proxy_mesh_counts("Cleaned proxy", proxy_verts, proxy_faces, log_prefix=log_prefix)
    return proxy_verts, proxy_faces


def _log_prefix(log_prefix: str) -> str:
    return f"{log_prefix.strip()} " if log_prefix else ""


def _log_proxy_mesh_counts(
    label: str,
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    log_prefix: str = _LOG_PREFIX,
) -> None:
    prefix = _log_prefix(log_prefix)
    print(f"{prefix}{label}: {len(verts)} verts / {len(faces)} tris")


def _meshlab_isotropic_remesh(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_edge_len = _target_edge_length_for_face_count(verts, faces, target_faces)
    out_v, out_f = _run_isotropic_remesh(
        verts,
        faces,
        target_edge_len=target_edge_len,
    )

    print(
        f"{_LOG_PREFIX} MeshLab isotropic-remeshed proxy (nondeterministic):"
        f" target={target_faces}, actual={len(out_v)} verts / {len(out_f)} tris,"
        f" target_edge_len={target_edge_len:.3g}"
    )
    return out_v, out_f


def _target_edge_length_for_face_count(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> float:
    tri_verts0 = verts[faces[:, 0]]
    tri_verts1 = verts[faces[:, 1]]
    tri_verts2 = verts[faces[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(
        np.cross(tri_verts1 - tri_verts0, tri_verts2 - tri_verts0),
        axis=1,
    )
    total_area = float(tri_areas.sum())
    if total_area <= 0.0:
        raise ValueError("Input mesh has zero surface area.")

    # Convert the requested face count to the edge length of an equivalent
    # equilateral-triangle tessellation over the same surface area.
    return float(
        np.sqrt((4.0 * total_area) / (np.sqrt(3.0) * float(target_faces)))
    )


def _run_isotropic_remesh(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    target_edge_len: float,
) -> tuple[np.ndarray, np.ndarray]:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts, faces))
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(float(target_edge_len)),
        iterations=10,
    )

    mesh = ms.current_mesh()
    out_v = np.array(mesh.vertex_matrix(), dtype=np.float32)
    out_f = np.array(mesh.face_matrix(), dtype=np.int32)

    return _sanitize_proxy_mesh(out_v, out_f, min_area=0.1)


def _cgal_isotropic_remesh(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from CGAL import CGAL_Polygon_mesh_processing
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    except ImportError as exc:
        raise RuntimeError(
            "The cgal-isotropic-remesh proxy mode requires the CGAL Python "
            "bindings. Install the examples extra or `pip install cgal`."
        ) from exc

    target_edge_len = _target_edge_length_for_face_count(verts, faces, target_faces)
    with tempfile.TemporaryDirectory(prefix="newton_bag_cgal_") as tmp_dir:
        input_path = f"{tmp_dir}/input.off"
        output_path = f"{tmp_dir}/output.off"
        _write_off_mesh(input_path, verts, faces)

        polyhedron = Polyhedron_3(input_path)
        facets = list(polyhedron.facets())
        CGAL_Polygon_mesh_processing.isotropic_remeshing(
            facets,
            float(target_edge_len),
            polyhedron,
            10,
        )
        polyhedron.write_to_file(output_path)
        out_v, out_f = _read_off_mesh(output_path)

    raw_count = len(out_f)
    out_v, out_f = _sanitize_proxy_mesh(out_v, out_f, min_area=0.1)
    print(
        f"{_LOG_PREFIX} CGAL isotropic-remeshed proxy:"
        f" target={target_faces}, raw={raw_count} tris,"
        f" actual={len(out_v)} verts / {len(out_f)} tris,"
        f" target_edge_len={target_edge_len:.3g}"
    )
    return out_v, out_f


def _write_off_mesh(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write("OFF\n")
        file.write(f"{len(verts)} {len(faces)} 0\n")
        for vertex in verts:
            file.write(f"{float(vertex[0]):.17g} {float(vertex[1]):.17g} {float(vertex[2]):.17g}\n")
        for face in faces:
            file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _read_off_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as file:
        tokens = [
            token
            for line in file
            for token in line.split("#", maxsplit=1)[0].split()
        ]

    if not tokens or tokens[0] != "OFF":
        raise ValueError(f"CGAL did not write an OFF mesh: {path}")
    if len(tokens) < 4:
        raise ValueError(f"CGAL wrote an incomplete OFF mesh: {path}")

    cursor = 1
    vertex_count = int(tokens[cursor])
    face_count = int(tokens[cursor + 1])
    cursor += 3

    verts = np.asarray(
        [
            [
                float(tokens[cursor + i * 3]),
                float(tokens[cursor + i * 3 + 1]),
                float(tokens[cursor + i * 3 + 2]),
            ]
            for i in range(vertex_count)
        ],
        dtype=np.float32,
    )
    cursor += vertex_count * 3

    faces: list[list[int]] = []
    for _ in range(face_count):
        face_size = int(tokens[cursor])
        cursor += 1
        face = [int(tokens[cursor + i]) for i in range(face_size)]
        cursor += face_size
        if face_size != 3:
            raise ValueError("CGAL isotropic remeshing produced a non-triangular OFF face.")
        faces.append(face)

    return verts, np.asarray(faces, dtype=np.int32)


def _surface_decimate(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import fast_simplification
    except ImportError as exc:
        raise RuntimeError(
            "The surface-decimate proxy mode requires fast_simplification"
        ) from exc

    out_v, out_f = fast_simplification.simplify(
        verts,
        faces,
        target_count=target_faces,
    )
    raw_count = len(out_f)
    out_v, out_f = _sanitize_proxy_mesh(out_v, out_f)
    print(
        f"{_LOG_PREFIX} Surface-decimated proxy:"
        f" target={target_faces}, raw={raw_count} tris,"
        f" actual={len(out_v)} verts / {len(out_f)} tris"
    )
    return out_v, out_f


def _qem_decimate(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=verts,
            face_matrix=faces,
        )
    )
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
    )
    mesh = ms.current_mesh()
    out_v = np.asarray(mesh.vertex_matrix(), dtype=np.float32)
    out_f = np.asarray(mesh.face_matrix(), dtype=np.int32).reshape(-1, 3)
    raw_count = len(out_f)
    out_v, out_f = _sanitize_proxy_mesh(out_v, out_f)
    print(
        f"{_LOG_PREFIX} QEM-decimated proxy:"
        f" target={target_faces}, raw={raw_count} tris,"
        f" actual={len(out_v)} verts / {len(out_f)} tris"
    )
    return out_v, out_f


def make_proxy_mesh_intersection_free(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    intersection_checker: Callable[[np.ndarray, np.ndarray], Sequence[tuple[int, int]]] | None = None,
    checker_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    edge_edge_min_distance: float = 0.0,
    min_area: float = 1.0e-12,
    max_meshlab_passes: int = 20,
    max_checker_passes: int = 12,
    max_edge_edge_passes: int = 12,
    log_prefix: str = _LOG_PREFIX,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove self-intersections from a proxy mesh for strict contact solvers.

    MeshLab's selector is used first because it can remove faces and close
    resulting holes in the same mesh state.  A caller may provide an additional
    checker, such as ppf-contact-solver's own self-intersection test, to remove
    any remaining pairs that MeshLab does not report.
    """
    out_v, out_f = _remove_meshlab_self_intersections(
        verts,
        faces,
        min_area=min_area,
        max_passes=max_meshlab_passes,
        log_prefix=log_prefix,
    )
    if intersection_checker is not None:
        out_v, out_f = _remove_checker_reported_intersections(
            out_v,
            out_f,
            intersection_checker,
            checker_transform=checker_transform,
            min_area=min_area,
            max_passes=max_checker_passes,
            log_prefix=log_prefix,
        )
    if edge_edge_min_distance > 0.0:
        out_v, out_f = _remove_edge_edge_proximity_intersections(
            out_v,
            out_f,
            min_distance=float(edge_edge_min_distance),
            min_area=min_area,
            max_passes=max_edge_edge_passes,
            log_prefix=log_prefix,
        )
        if intersection_checker is not None:
            out_v, out_f = _remove_checker_reported_intersections(
                out_v,
                out_f,
                intersection_checker,
                checker_transform=checker_transform,
                min_area=min_area,
                max_passes=max_checker_passes,
                log_prefix=log_prefix,
            )
    return out_v, out_f


def _remove_meshlab_self_intersections(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    min_area: float,
    max_passes: int,
    log_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts, faces))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    total_selected = 0
    for _ in range(max_passes):
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        try:
            ms.compute_selection_by_self_intersections_per_face()
        except Exception:
            break
        selected_faces = ms.current_mesh().selected_face_number()
        if selected_faces == 0:
            break
        total_selected += selected_faces
        ms.meshing_remove_selected_faces()
        try:
            ms.meshing_close_holes(maxholesize=20)
        except Exception:
            pass

    mesh = ms.current_mesh()
    out_v = np.asarray(mesh.vertex_matrix(), dtype=np.float32)
    out_f = np.asarray(mesh.face_matrix(), dtype=np.int32).reshape(-1, 3)
    out_v, out_f = _sanitize_proxy_mesh(out_v, out_f, min_area=min_area)
    if total_selected:
        prefix = f"{log_prefix} " if log_prefix else ""
        print(
            f"{prefix}Removed MeshLab-selected self-intersecting proxy faces "
            f"({total_selected} selections across cleanup passes)."
        )
    return out_v, out_f


def _remove_checker_reported_intersections(
    verts: np.ndarray,
    faces: np.ndarray,
    intersection_checker: Callable[[np.ndarray, np.ndarray], Sequence[tuple[int, int]]],
    *,
    checker_transform: Callable[[np.ndarray], np.ndarray] | None,
    min_area: float,
    max_passes: int,
    log_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    out_v, out_f = _sanitize_proxy_mesh(verts, faces, min_area=min_area)
    total_removed = 0
    for _ in range(max_passes):
        check_v = checker_transform(out_v) if checker_transform is not None else out_v
        pairs = intersection_checker(
            np.ascontiguousarray(check_v, dtype=np.float64),
            np.ascontiguousarray(out_f, dtype=np.int32),
        )
        tri_pairs = [(int(a), int(b)) for a, b in pairs if a >= 0 and b >= 0]
        if not tri_pairs:
            break

        counts = np.zeros(len(out_f), dtype=np.int32)
        for a, b in tri_pairs:
            counts[a] += 1
            counts[b] += 1
        areas = _triangle_areas(out_v, out_f)

        remove: set[int] = set()
        for a, b in tri_pairs:
            if a in remove or b in remove:
                continue
            if counts[a] > counts[b]:
                drop = a
            elif counts[b] > counts[a]:
                drop = b
            else:
                drop = a if areas[a] <= areas[b] else b
            remove.add(drop)

        keep = np.ones(len(out_f), dtype=bool)
        keep[list(remove)] = False
        out_f = out_f[keep]
        total_removed += len(remove)
        out_v, out_f = _sanitize_proxy_mesh(out_v, out_f, min_area=min_area)
    else:
        raise RuntimeError("Unable to remove all reported proxy self-intersections.")

    if total_removed:
        prefix = f"{log_prefix} " if log_prefix else ""
        print(
            f"{prefix}Removed {total_removed} additional proxy faces reported by "
            "the strict self-intersection checker."
        )
    return out_v, out_f


def _remove_edge_edge_proximity_intersections(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    min_distance: float,
    min_area: float,
    max_passes: int,
    log_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    out_v, out_f = _sanitize_proxy_mesh(verts, faces, min_area=min_area)
    total_removed = 0
    for _ in range(max_passes):
        conflicts = _find_edge_edge_proximity_conflicts(out_v, out_f, min_distance)
        if not conflicts:
            break

        face_conflict_count = np.zeros(len(out_f), dtype=np.int32)
        for edge_a, edge_b in conflicts:
            for face_id in edge_a[2]:
                face_conflict_count[face_id] += 1
            for face_id in edge_b[2]:
                face_conflict_count[face_id] += 1

        areas = _triangle_areas(out_v, out_f)
        remove: set[int] = set()
        for edge_a, edge_b in conflicts:
            candidates = set(edge_a[2]) | set(edge_b[2])
            candidates.difference_update(remove)
            if not candidates:
                continue
            drop = max(
                candidates,
                key=lambda face_id: (face_conflict_count[face_id], -areas[face_id]),
            )
            remove.add(drop)

        keep = np.ones(len(out_f), dtype=bool)
        keep[list(remove)] = False
        out_f = out_f[keep]
        total_removed += len(remove)
        out_v, out_f = _sanitize_proxy_mesh(out_v, out_f, min_area=min_area)
    else:
        raise RuntimeError("Unable to remove all proxy edge-edge proximity intersections.")

    if total_removed:
        prefix = f"{log_prefix} " if log_prefix else ""
        print(
            f"{prefix}Removed {total_removed} proxy faces to satisfy "
            f"edge-edge separation >= {min_distance:.3g}."
        )
    return out_v, out_f


def _find_edge_edge_proximity_conflicts(
    verts: np.ndarray,
    faces: np.ndarray,
    min_distance: float,
) -> list[tuple[tuple[int, int, tuple[int, ...]], tuple[int, int, tuple[int, ...]]]]:
    edges = _extract_edges_with_faces(faces)
    if len(edges) < 2:
        return []

    edge_indices = np.array([(a, b) for a, b, _ in edges], dtype=np.int32)
    p0 = verts[edge_indices[:, 0]]
    p1 = verts[edge_indices[:, 1]]
    mins = np.minimum(p0, p1) - min_distance
    maxs = np.maximum(p0, p1) + min_distance
    order = np.argsort(mins[:, 0])
    min_dist2 = min_distance * min_distance
    conflicts = []

    for order_pos, edge_id in enumerate(order):
        max_x = maxs[edge_id, 0]
        for other_id in order[order_pos + 1:]:
            if mins[other_id, 0] > max_x:
                break
            if np.any(mins[edge_id] > maxs[other_id]) or np.any(mins[other_id] > maxs[edge_id]):
                continue
            edge_a = edges[edge_id]
            edge_b = edges[other_id]
            dist2 = _native_style_edge_edge_distance2(
                verts,
                edge_a[0],
                edge_a[1],
                edge_b[0],
                edge_b[1],
            )
            if dist2 < min_dist2:
                conflicts.append((edge_a, edge_b))
    return conflicts


def _extract_edges_with_faces(faces: np.ndarray) -> list[tuple[int, int, tuple[int, ...]]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, tri in enumerate(faces):
        for i0, i1 in ((0, 1), (1, 2), (2, 0)):
            a = int(tri[i0])
            b = int(tri[i1])
            edge = (a, b) if a < b else (b, a)
            edge_faces.setdefault(edge, []).append(face_id)
    return [(a, b, tuple(face_ids)) for (a, b), face_ids in edge_faces.items()]


def _native_style_edge_edge_distance2(
    verts: np.ndarray,
    a: int,
    b: int,
    c: int,
    d: int,
) -> float:
    shared = set((a, b)) & set((c, d))
    if shared:
        shared_id = shared.pop()
        other_ab = b if a == shared_id else a
        other_cd = d if c == shared_id else c
        p_shared = verts[shared_id]
        p_ab = verts[other_ab]
        p_cd = verts[other_cd]
        return min(
            _point_segment_distance2(p_ab, p_shared, p_cd),
            _point_segment_distance2(p_cd, p_shared, p_ab),
        )
    return _segment_segment_distance2(verts[a], verts[b], verts[c], verts[d])


def _point_segment_distance2(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    edge = seg_b - seg_a
    denom = float(edge @ edge)
    if denom <= 0.0:
        delta = point - seg_a
        return float(delta @ delta)
    t = float((point - seg_a) @ edge) / denom
    t = max(0.0, min(1.0, t))
    delta = point - (seg_a + t * edge)
    return float(delta @ delta)


def _segment_segment_distance2(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> float:
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(u @ u)
    b = float(u @ v)
    c = float(v @ v)
    d = float(u @ w)
    e = float(v @ w)
    denom = a * c - b * b
    eps = 1.0e-20
    s_denom = denom
    t_denom = denom

    if denom < eps:
        s_num = 0.0
        s_denom = 1.0
        t_num = e
        t_denom = c
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d
        if s_num < 0.0:
            s_num = 0.0
            t_num = e
            t_denom = c
        elif s_num > s_denom:
            s_num = s_denom
            t_num = e + b
            t_denom = c

    if t_num < 0.0:
        t_num = 0.0
        if -d < 0.0:
            s_num = 0.0
        elif -d > a:
            s_num = s_denom
        else:
            s_num = -d
            s_denom = a
    elif t_num > t_denom:
        t_num = t_denom
        if -d + b < 0.0:
            s_num = 0.0
        elif -d + b > a:
            s_num = s_denom
        else:
            s_num = -d + b
            s_denom = a

    s = 0.0 if abs(s_num) < eps else s_num / s_denom
    t = 0.0 if abs(t_num) < eps else t_num / t_denom
    delta = w + s * u - t * v
    return float(delta @ delta)


def _sanitize_proxy_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    min_area: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    out_v = np.asarray(verts, dtype=np.float32)
    out_f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    unique_vertex_ids = (
        (out_f[:, 0] != out_f[:, 1])
        & (out_f[:, 1] != out_f[:, 2])
        & (out_f[:, 2] != out_f[:, 0])
    )
    out_f = out_f[unique_vertex_ids]

    areas = _triangle_areas(out_v, out_f)
    keep = areas > min_area
    out_f = out_f[keep]

    tri_keys = np.sort(out_f, axis=1)
    _, unique_ids = np.unique(tri_keys, axis=0, return_index=True)
    out_f = out_f[np.sort(unique_ids)]

    while True:
        tri_areas = _triangle_areas(out_v, out_f)

        edge_faces: dict[tuple[int, int], list[int]] = {}
        for face_id, tri in enumerate(out_f):
            for i0, i1 in ((0, 1), (1, 2), (2, 0)):
                a = int(tri[i0])
                b = int(tri[i1])
                edge = (a, b) if a < b else (b, a)
                edge_faces.setdefault(edge, []).append(face_id)

        remove_faces = set()
        for face_ids in edge_faces.values():
            if len(face_ids) > 2:
                drop_face = min(
                    face_ids,
                    key=lambda face_id: tri_areas[face_id],
                )
                remove_faces.add(drop_face)

        if not remove_faces:
            break

        keep_faces = np.ones(len(out_f), dtype=bool)
        keep_faces[list(remove_faces)] = False
        out_f = out_f[keep_faces]

    used = np.unique(out_f)
    remap = np.full(len(out_v), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    out_v = out_v[used]
    out_f = remap[out_f]

    return out_v, out_f


def _triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(
        np.cross(
            verts[faces[:, 1]] - verts[faces[:, 0]],
            verts[faces[:, 2]] - verts[faces[:, 0]],
        ),
        axis=1,
    )
