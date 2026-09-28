# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unchanged metric definitions from the May 26 rigidity sweep."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshMetricsTopology:
    faces: np.ndarray
    edges: np.ndarray
    bend_pairs: np.ndarray


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in faces:
        edges.add(tuple(sorted((int(a), int(b)))))
        edges.add(tuple(sorted((int(b), int(c)))))
        edges.add(tuple(sorted((int(c), int(a)))))
    return np.array(sorted(edges), dtype=np.int32)


def _bend_pairs(faces: np.ndarray) -> np.ndarray:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for edge in (a, b), (b, c), (c, a):
            edge_to_faces.setdefault(tuple(sorted((int(edge[0]), int(edge[1])))), []).append(face_index)

    pairs = [face_indices[:2] for face_indices in edge_to_faces.values() if len(face_indices) == 2]
    return np.array(pairs, dtype=np.int32)


def _triangle_angles(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = points[faces]
    lengths = np.stack(
        [
            np.linalg.norm(tris[:, 1] - tris[:, 2], axis=1),
            np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1),
            np.linalg.norm(tris[:, 0] - tris[:, 1], axis=1),
        ],
        axis=1,
    )
    a = lengths[:, 0]
    b = lengths[:, 1]
    c = lengths[:, 2]
    eps = 1.0e-12
    angles = np.empty((faces.shape[0], 3), dtype=np.float64)
    angles[:, 0] = np.arccos(np.clip((b * b + c * c - a * a) / np.maximum(2.0 * b * c, eps), -1.0, 1.0))
    angles[:, 1] = np.arccos(np.clip((c * c + a * a - b * b) / np.maximum(2.0 * c * a, eps), -1.0, 1.0))
    angles[:, 2] = np.arccos(np.clip((a * a + b * b - c * c) / np.maximum(2.0 * a * b, eps), -1.0, 1.0))
    return angles


def _face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = points[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0
    return normals


def _dihedral_angles(points: np.ndarray, faces: np.ndarray, bend_pairs: np.ndarray) -> np.ndarray:
    normals = _face_normals(points, faces)
    paired = normals[bend_pairs]
    dots = np.sum(paired[:, 0] * paired[:, 1], axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def _build_topology(faces: np.ndarray) -> MeshMetricsTopology:
    return MeshMetricsTopology(faces=faces, edges=_unique_edges(faces), bend_pairs=_bend_pairs(faces))


def _measure(
    points: np.ndarray,
    topology: MeshMetricsTopology,
    rest_edge_lengths: np.ndarray,
    rest_angles: np.ndarray,
    rest_dihedrals: np.ndarray,
) -> tuple[float, float, float]:
    edge_lengths = np.linalg.norm(points[topology.edges[:, 1]] - points[topology.edges[:, 0]], axis=1)
    stretch = np.sqrt(np.mean(((edge_lengths - rest_edge_lengths) / np.maximum(rest_edge_lengths, 1.0e-12)) ** 2))

    angles = _triangle_angles(points, topology.faces)
    shear = np.sqrt(np.mean((angles - rest_angles) ** 2))

    dihedrals = _dihedral_angles(points, topology.faces, topology.bend_pairs)
    bend = np.sqrt(np.mean((dihedrals - rest_dihedrals) ** 2))
    return float(stretch), float(shear), float(bend)
