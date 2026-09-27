# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body surface samples for contact: exposed-face centroids and outward normals.

This module is experimental and may change without API compatibility. It
implements section 3.1 of ``notes/contact-design-20260927.md``: one contact
sample per exposed hexahedral face, located at the face centroid with the
outward unit normal built from the two face diagonals. Both quantities are
differentiable in the corner positions so contact forces reach the corners.

Face corner quads are derived from :attr:`VoxelGridData.cell_corner_indices`
in z-fast local corner order (bits x, y, z with z fastest) for the six
material faces in ``(-x, +x, -y, +y, -z, +z)`` order. The corner order per
face is fixed so that ``normalize((x2 - x0) x (x3 - x1))`` points outward at
rest; :func:`exposed_face_samples` verifies this numerically and refuses to
return samples whose rest normal points inward.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor  # noqa: TID253

if TYPE_CHECKING:
    from .data import VoxelGridData

__all__ = [
    "FACE_LOCAL_CORNERS",
    "FACE_NAMES",
    "FaceSamples",
    "exposed_face_samples",
    "sample_normals",
    "sample_points",
]

FACE_NAMES = ("-x", "+x", "-y", "+y", "-z", "+z")
"""Material face names in the order used by :attr:`VoxelGridData.cell_neighbors`."""

FACE_LOCAL_CORNERS = (
    (0, 1, 3, 2),  # -x
    (4, 6, 7, 5),  # +x
    (0, 4, 5, 1),  # -y
    (2, 3, 7, 6),  # +y
    (0, 2, 6, 4),  # -z
    (1, 5, 7, 3),  # +z
)
"""Local z-fast corner ids of each material face, cyclic and counter-clockwise seen from outside.

With quad corners ``x0..x3`` in this order the diagonal cross product
``(x2 - x0) x (x3 - x1)`` points out of the cell on the rest grid.
"""

_DEFAULT_EPS = 1e-12


class FaceSamples(NamedTuple):
    """Static description of one contact sample per exposed face.

    Experimental: fields may change without notice. All tensors live on the
    CPU; move them to the positions' device before calling
    :func:`sample_points` or :func:`sample_normals`.
    """

    cell_index: Tensor
    """Owning cell of each sample, int64, shape [S]."""

    face_index: Tensor
    """Material face id 0..5 in ``(-x, +x, -y, +y, -z, +z)`` order, int64, shape [S]."""

    corners: Tensor
    """Global corner ids of each face quad, int64, shape [S, 4], ordered so the diagonal normal points outward at rest."""

    rest_normals: Tensor
    """Outward unit face normals on the rest grid, float32, shape [S, 3]."""


def _require_tensor(name: str, value) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _check_positions_and_corners(positions: Tensor, corners: Tensor) -> None:
    """Validate the shared ``positions [B, P, 3]`` and ``corners [S, 4]`` arguments."""
    _require_tensor("positions", positions)
    _require_tensor("corners", corners)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B, P, 3]")
    if not positions.is_floating_point():
        raise TypeError("positions must have a floating dtype")
    if corners.ndim != 2 or corners.shape[-1] != 4:
        raise ValueError("corners must have shape [S, 4]")
    if corners.dtype != torch.int64:
        raise TypeError("corners must have dtype int64")
    if corners.device != positions.device:
        raise ValueError("corners must be on the same device as positions")


def _quad_positions(positions: Tensor, corners: Tensor) -> Tensor:
    """Gather the four corner positions of every sample, shape [B, S, 4, 3]."""
    return positions[:, corners]


def sample_points(positions: Tensor, corners: Tensor) -> Tensor:
    """Return the centroid of each face quad, the mean of its four corners.

    Args:
        positions: Current corner positions [m], shape [B, P, 3].
        corners: Global corner ids per sample from :class:`FaceSamples`, int64, shape [S, 4].

    Returns:
        Sample positions [m], shape [B, S, 3]; differentiable in ``positions``.

    Raises:
        ValueError: If the shapes or devices are inconsistent.
        TypeError: If the inputs are not tensors of the expected dtype family.
    """
    _check_positions_and_corners(positions, corners)
    return _quad_positions(positions, corners).mean(dim=2)


def sample_normals(
    positions: Tensor,
    corners: Tensor,
    rest_normals: Tensor,
    *,
    eps: float = _DEFAULT_EPS,
) -> Tensor:
    """Return the unit normal of each face quad from its two diagonals.

    The normal is ``normalize((x2 - x0) x (x3 - x1))``; the corner order from
    :func:`exposed_face_samples` makes it point outward. Faces whose diagonal
    cross product has a norm below ``eps`` fall back to ``rest_normals`` and
    receive a zero (finite) gradient.

    Args:
        positions: Current corner positions [m], shape [B, P, 3].
        corners: Global corner ids per sample from :class:`FaceSamples`, int64, shape [S, 4].
        rest_normals: Fallback unit normals for degenerate faces, shape [S, 3]; cast to the dtype of ``positions``.
        eps: Degeneracy threshold on the diagonal cross-product norm [m^2].

    Returns:
        Unit normals, shape [B, S, 3]; differentiable in ``positions``.

    Raises:
        ValueError: If the shapes or devices are inconsistent or ``eps`` is not positive.
        TypeError: If the inputs are not tensors of the expected dtype family.
    """
    _check_positions_and_corners(positions, corners)
    _require_tensor("rest_normals", rest_normals)
    if rest_normals.shape != (corners.shape[0], 3):
        raise ValueError("rest_normals must have shape [S, 3] matching corners")
    if not rest_normals.is_floating_point():
        raise TypeError("rest_normals must have a floating dtype")
    if rest_normals.device != positions.device:
        raise ValueError("rest_normals must be on the same device as positions")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite number")

    quad = _quad_positions(positions, corners)
    cross = torch.linalg.cross(quad[:, :, 2] - quad[:, :, 0], quad[:, :, 3] - quad[:, :, 1], dim=-1)
    norm_sq = cross.square().sum(dim=-1, keepdim=True)
    degenerate = norm_sq < float(eps) ** 2
    # Divide by one on degenerate faces so the unselected branch has a finite gradient.
    safe_norm_sq = torch.where(degenerate, torch.ones_like(norm_sq), norm_sq)
    unit = cross * torch.rsqrt(safe_norm_sq)
    fallback = rest_normals.to(dtype=positions.dtype).unsqueeze(0).expand_as(unit)
    return torch.where(degenerate, fallback, unit)


def exposed_face_samples(rest: VoxelGridData) -> FaceSamples:
    """Build one contact sample per exposed face of the rest grid.

    Samples are ordered by cell index, then by material face index. The rest
    normals are computed with the same diagonal formula as
    :func:`sample_normals` and checked to point away from the owning cell's
    rest center, so a wrong corner order raises instead of shipping silently.

    Args:
        rest: Rest grid with z-fast ``cell_corner_indices`` and material-order ``cell_neighbors``.

    Returns:
        CPU tensors describing the S exposed-face samples.

    Raises:
        ValueError: If the grid arrays have inconsistent shapes or indices, a
            rest face is degenerate, or a rest normal does not point outward.
    """
    corner_indices = np.asarray(rest.cell_corner_indices)
    neighbors = np.asarray(rest.cell_neighbors)
    rest_positions = np.asarray(rest.corner_rest_positions, dtype=np.float64)
    if corner_indices.ndim != 2 or corner_indices.shape[1] != 8 or not np.issubdtype(corner_indices.dtype, np.integer):
        raise ValueError("rest.cell_corner_indices must be an integer array of shape [C, 8]")
    if neighbors.shape != (corner_indices.shape[0], 6):
        raise ValueError("rest.cell_neighbors must have shape [C, 6]")
    if rest_positions.ndim != 2 or rest_positions.shape[1] != 3 or not np.isfinite(rest_positions).all():
        raise ValueError("rest.corner_rest_positions must be a finite array of shape [P, 3]")
    if corner_indices.size and (corner_indices.min() < 0 or corner_indices.max() >= rest_positions.shape[0]):
        raise ValueError("rest.cell_corner_indices must index into rest.corner_rest_positions")

    cell_index, face_index = np.nonzero(neighbors == -1)
    cell_index = cell_index.astype(np.int64)
    face_index = face_index.astype(np.int64)
    local_corners = np.asarray(FACE_LOCAL_CORNERS, dtype=np.int64)[face_index]
    corners = corner_indices.astype(np.int64)[cell_index[:, None], local_corners]

    quad = rest_positions[corners]
    cross = np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
    norm = np.linalg.norm(cross, axis=-1)
    if np.any(norm < _DEFAULT_EPS):
        raise ValueError("rest grid has a degenerate exposed face")
    normals = cross / norm[:, None]

    cell_centers = rest_positions[corner_indices[cell_index]].mean(axis=1)
    outward = np.einsum("ij,ij->i", normals, quad.mean(axis=1) - cell_centers)
    if np.any(outward <= 0.0):
        bad = int(np.argmin(outward))
        raise ValueError(
            f"rest normal of face {FACE_NAMES[face_index[bad]]} on cell {cell_index[bad]} points inward; "
            "cell_corner_indices must use z-fast local corner order"
        )

    return FaceSamples(
        cell_index=torch.from_numpy(cell_index),
        face_index=torch.from_numpy(face_index),
        corners=torch.from_numpy(np.ascontiguousarray(corners)),
        rest_normals=torch.from_numpy(normals.astype(np.float32)),
    )
