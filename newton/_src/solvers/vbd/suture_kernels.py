# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Suturing in SolverVBD: tunnel point rows, the needle state machine and the contact gate.

A suture needle (see :meth:`newton.ModelBuilder.add_suture_needle`) opens a virtual tunnel through a
tetrahedral soft body: a chain of tunnel points, each a barycentric point of one tet. Every point
carries a hard bilateral row in the two directions transverse to the rod tangent and the capped ALM
friction of body-body contact along the tangent. The design is in ``notes/vbd-suture/vbd-suture.md``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

# Needle states.
SUTURE_OUTSIDE = wp.constant(0)
SUTURE_PIERCING = wp.constant(1)
SUTURE_THROUGH = wp.constant(2)

# Tunnel states.
TUNNEL_UNUSED = wp.constant(0)
TUNNEL_PIERCING = wp.constant(1)
TUNNEL_THROUGH = wp.constant(2)
TUNNEL_FINISHED = wp.constant(3)

# The face opposite local vertex i of a tet, as local indices.
_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def build_tet_neighbors(tet_indices: np.ndarray) -> np.ndarray:
    """Return the tet across the face opposite each local vertex, or -1 on the boundary.

    Args:
        tet_indices: Tet vertex indices, shape ``[tet_count, 4]``.

    Returns:
        int32 array of shape ``[tet_count, 4]``; entry ``[t, i]`` is the tet that shares the face
        opposite vertex ``i`` of tet ``t``.
    """
    tets = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)
    count = tets.shape[0]
    neighbors = np.full((count, 4), -1, dtype=np.int32)
    if count == 0:
        return neighbors
    faces = np.empty((count * 4, 3), dtype=np.int64)
    for i, (a, b, c) in enumerate(_TET_FACES):
        faces[i::4] = np.sort(tets[:, [a, b, c]], axis=1)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    sorted_faces = faces[order]
    same = np.all(sorted_faces[1:] == sorted_faces[:-1], axis=1)
    first = order[:-1][same]
    second = order[1:][same]
    neighbors[first // 4, first % 4] = second // 4
    neighbors[second // 4, second % 4] = first // 4
    return neighbors
