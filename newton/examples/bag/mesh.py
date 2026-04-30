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

import numpy as np

import newton.examples

DEFAULT_PROXY_MODE = "isotropic-remesh"
PROXY_MODES = (DEFAULT_PROXY_MODE, "surface-decimate", "qem-decimate")


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
        help="Target proxy face count for cloth simulation",
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
) -> tuple[np.ndarray, np.ndarray]:
    """Create a lower-resolution bag proxy mesh for VBD cloth simulation."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    target_faces = max(int(target_faces), 1)

    if proxy_mode == DEFAULT_PROXY_MODE:
        return _isotropic_remesh(verts, faces, target_faces)
    if proxy_mode == "surface-decimate":
        return _surface_decimate(verts, faces, target_faces)
    if proxy_mode == "qem-decimate":
        return _qem_decimate(verts, faces, target_faces)

    raise ValueError(f"Unknown proxy mode: {proxy_mode}")


def _isotropic_remesh(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    target_edge_len = float(
        np.sqrt((4.0 * total_area) / (np.sqrt(3.0) * float(target_faces)))
    )
    out_v, out_f = _run_isotropic_remesh(
        verts,
        faces,
        target_edge_len=target_edge_len,
    )

    print(
        "Isotropic-remeshed proxy:"
        f" target={target_faces}, actual={len(out_f)},"
        f" target_edge_len={target_edge_len:.3g}"
    )
    return out_v, out_f


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
        "Surface-decimated proxy:"
        f" target={target_faces}, raw={raw_count}, actual={len(out_f)}"
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
        "QEM-decimated proxy:"
        f" target={target_faces}, raw={raw_count}, actual={len(out_f)}"
    )
    return out_v, out_f


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

    areas = np.array([
        0.5 * np.linalg.norm(
            np.cross(
                out_v[tri[1]] - out_v[tri[0]],
                out_v[tri[2]] - out_v[tri[0]],
            )
        )
        for tri in out_f
    ])
    keep = areas > min_area
    out_f = out_f[keep]

    tri_keys = np.sort(out_f, axis=1)
    _, unique_ids = np.unique(tri_keys, axis=0, return_index=True)
    out_f = out_f[np.sort(unique_ids)]

    while True:
        tri_areas = np.array([
            0.5 * np.linalg.norm(
                np.cross(
                    out_v[tri[1]] - out_v[tri[0]],
                    out_v[tri[2]] - out_v[tri[0]],
                )
            )
            for tri in out_f
        ])

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
