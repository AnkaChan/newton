# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closest proper rotation frames with a deterministic clamped-face tie-break.

Each cell frame is the right-handed rotation closest to the cell-center
deformation gradient F in the Frobenius norm. Inverted (det F < 0) and
collapsed cells are accepted: a handedness correction acts on the smallest
singular direction, so ``R = U @ diag(1, 1, det(U @ Vh)) @ Vh``. Equally close
rotations (rank-deficient F, or inverted F with equal middle and smallest
singular values) are resolved with a reference frame built from three ordered
prescribed corners of the clamped face. The reference rotates with the whole
problem, so the resolved frame does too; it only breaks ties and never replaces
a uniquely closest rotation.

Frames are computed without differentiation and returned detached. F itself is
never modified; callers compute the local axes ``A = R^T F`` so ``R @ A = F``
holds for every cell, including inverted ones. This tie rule is specific to the
clamped prototype (three noncollinear prescribed corners).
"""

import math
from typing import NamedTuple

import numpy as np
import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor  # noqa: TID253

__all__ = ["FrameResult", "closest_proper_rotations", "reference_rotation", "select_reference_corners"]

DEGENERATE_NORM = 1e-12
"""Absolute norm [m] below which reference edges or normals are degenerate."""

COLLINEAR_AREA_RATIO = 1e-12
"""Area over squared base length below which prescribed corners count as collinear."""

CORNER_TIE_RATIO = 1e-9
"""Relative tolerance for equal distances or areas in corner selection; ties take the smallest ID."""


class FrameResult(NamedTuple):
    """Detached per-cell frames from :func:`closest_proper_rotations`.

    Attributes:
        frames: Proper rotations R with world columns, shape [B, C, 3, 3].
        tie_mask: Boolean [B, C]; True where the closest rotation was ambiguous
            within ``tie_tolerance``. With a reference these are exactly the
            cells whose frame came from the tie-break; without one the mask
            still reports the detected ties although the plain formula was kept.
        singular_values: Singular values of the unmodified F, descending, shape [B, C, 3].
    """

    frames: Tensor
    tie_mask: Tensor
    singular_values: Tensor


def select_reference_corners(rest_positions: np.ndarray, fixed_indices) -> np.ndarray | None:
    """Return three ordered corner IDs from the prescribed set that span the clamped face, or None.

    The selection depends only on rest geometry and prescribed IDs, so it is
    fixed for a prototype: ``p0`` is the prescribed corner with the
    lexicographically smallest rest position (x, then y, then z); ``p1`` is the
    prescribed corner farthest from ``p0``; ``p2`` is the prescribed corner
    maximizing the parallelogram area ``|(p1 - p0) x (p2 - p0)|``. Distances or
    areas equal within :data:`CORNER_TIE_RATIO` relative resolve to the
    smallest corner ID. Duplicate prescribed IDs are ignored.

    Args:
        rest_positions: Rest corner positions [m], shape [corner_count, 3].
        fixed_indices: Prescribed corner IDs (integer array-like or CPU tensor).

    Returns:
        ``int64`` array ``[p0, p1, p2]``, or None when fewer than three
        noncollinear prescribed corners exist (largest area below
        ``COLLINEAR_AREA_RATIO * |p1 - p0|^2``).

    Raises:
        ValueError: Malformed positions or out-of-range, non-integer indices.
    """
    positions = np.asarray(rest_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("rest_positions must have shape [corner_count, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("rest_positions must contain only finite values")
    if isinstance(fixed_indices, Tensor):
        fixed_indices = fixed_indices.detach().cpu().numpy()
    fixed = np.asarray(fixed_indices)
    if fixed.ndim != 1 or (fixed.size and fixed.dtype.kind not in "iu"):
        raise ValueError("fixed_indices must be a one-dimensional integer array")
    if fixed.size and (fixed.min() < 0 or fixed.max() >= len(positions)):
        raise ValueError("fixed_indices must be in range for rest_positions")
    fixed = np.unique(fixed.astype(np.int64))
    if len(fixed) < 3:
        return None
    pinned = positions[fixed]
    first = int(np.lexsort((pinned[:, 2], pinned[:, 1], pinned[:, 0]))[0])
    distances = np.linalg.norm(pinned - pinned[first], axis=1)
    farthest = float(distances.max())
    if farthest <= 0.0:
        return None
    second = int(np.flatnonzero(distances >= farthest * (1.0 - CORNER_TIE_RATIO))[0])
    areas = np.linalg.norm(np.cross(pinned[second] - pinned[first], pinned - pinned[first]), axis=1)
    largest = float(areas.max())
    if largest < COLLINEAR_AREA_RATIO * farthest**2:
        return None
    third = int(np.flatnonzero(areas >= largest * (1.0 - CORNER_TIE_RATIO))[0])
    return fixed[[first, second, third]].astype(np.int64)


def reference_rotation(positions: Tensor, corner_indices: Tensor) -> Tensor:
    """Build the proper reference frame from the current positions of three ordered corners.

    With ``x0, x1, x2`` the current positions of the ordered corners,
    ``e1 = normalize(x1 - x0)``, ``n = normalize(e1 x (x2 - x0))`` and
    ``e2 = n x e1``; the frame has columns ``[e1, e2, n]``. Rotating and
    translating all positions rotates the frame by the same rotation.

    Args:
        positions: Current corner positions [m], shape [B, P, 3], floating point.
        corner_indices: Three distinct corner IDs, shape [3] (integer tensor or array-like).

    Returns:
        Detached proper rotations, shape [B, 3, 3], on the positions' dtype and device.

    Raises:
        TypeError: ``positions`` is not a tensor.
        ValueError: Malformed inputs, nonfinite corner positions, or degenerate
            corners (edge or normal norm below :data:`DEGENERATE_NORM`).
    """
    if not isinstance(positions, Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim != 3 or positions.shape[-1] != 3 or not positions.is_floating_point():
        raise ValueError("positions must be a floating-point tensor of shape [B, P, 3]")
    indices = torch.as_tensor(corner_indices)
    if indices.shape != (3,) or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("corner_indices must be three integer corner IDs")
    indices = indices.to(torch.long).cpu()
    if len(torch.unique(indices)) != 3 or bool(indices.min() < 0) or bool(indices.max() >= positions.shape[1]):
        raise ValueError("corner_indices must be three distinct in-range corner IDs")
    with torch.no_grad():
        corners = positions.detach()[:, indices.to(positions.device)]
        if not torch.isfinite(corners).all():
            raise ValueError("reference corner positions must be finite")
        edge = corners[:, 1] - corners[:, 0]
        edge_norm = torch.linalg.vector_norm(edge, dim=-1, keepdim=True)
        if bool((edge_norm < DEGENERATE_NORM).any()):
            raise ValueError("reference corners 0 and 1 coincide")
        first = edge / edge_norm
        normal = torch.linalg.cross(first, corners[:, 2] - corners[:, 0])
        normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        if bool((normal_norm < DEGENERATE_NORM).any()):
            raise ValueError("reference corners are collinear")
        normal = normal / normal_norm
        second = torch.linalg.cross(normal, first)
        return torch.stack((first, second, normal), dim=-1)


def _proper_rotation(left: Tensor, right_transpose: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``U @ diag(1, 1, d) @ Vh`` and ``d = sign(det(U @ Vh))`` with zero treated as +1."""
    determinant = torch.linalg.det(left @ right_transpose)
    orientation = torch.where(determinant < 0, -1.0, 1.0).to(left.dtype)
    scale = torch.ones_like(left[..., 0, :])
    scale[..., 2] = orientation
    return (left * scale[..., None, :]) @ right_transpose, orientation


def closest_proper_rotations(
    deformation: Tensor, reference: Tensor | None = None, *, tie_tolerance: float = 1e-4
) -> FrameResult:
    """Return the closest proper rotation to each F with a deterministic tie-break.

    For ``U, s, Vh = svd(F)`` with descending ``s`` and ``d = sign(det(U @ Vh))``
    (zero counts as +1), the closest proper rotation is
    ``R = U @ diag(1, 1, d) @ Vh``: for inverted F the handedness flip acts on
    the smallest singular direction. Its Frobenius margin over the next-closest
    proper rotation is ``4 * gap`` with ``gap = s2 + s3`` when ``d > 0`` and
    ``gap = s2 - s3`` when ``d < 0``. Cells with
    ``gap <= tie_tolerance * max(s1, 1)`` are ties (default 1e-4 relative to
    the largest singular value, absolute below unit stretch).

    For tie cells with a reference, the rotation is recomputed from the
    perturbed matrix ``G = F + eps * R_ref`` with ``eps = tie_tolerance * max(s1, 1)``
    using the same formula and no further tie handling. To first order this
    selects, among the equally close rotations, the one closest to ``R_ref``:
    for rank-one F it maps the leading right singular vector to the leading
    left one and rotates about that axis to best match the reference; for
    ``F = 0`` it returns ``R_ref``. The perturbed problem is conditioned like
    ``1 / tie_tolerance``, so the perturbed matrix is formed and solved in
    float64 for the tie cells only and cast back; only tie cells are ever modified. Without a reference tie
    cells keep the plain formula (documented fallback for prototypes without
    three noncollinear prescribed corners).

    Caveat: the frame is discontinuous across the tie threshold and, for tie
    cells, jumps with the reference. This is a coordinate choice for the
    network; the local axes ``R^T F`` always reconstruct F exactly.

    Args:
        deformation: Cell-center deformation gradients, shape [B, C, 3, 3], float32 or float64.
        reference: Proper reference rotations ``R_ref``, shape [B, 3, 3], same dtype and
            device as ``deformation`` (broadcast over cells), or None.
        tie_tolerance: Nonnegative relative tie threshold; zero disables the tie-break.

    Returns:
        Detached frames, tie mask and singular values of the unmodified F.

    Raises:
        TypeError: Non-tensor inputs or a non-numeric tolerance.
        ValueError: Wrong shapes, dtype or device mismatch, nonfinite values,
            a reference that is not a proper rotation, or an invalid tolerance.
    """
    if not isinstance(deformation, Tensor):
        raise TypeError("deformation must be a torch.Tensor")
    if isinstance(tie_tolerance, bool) or not isinstance(tie_tolerance, (int, float)):
        raise TypeError("tie_tolerance must be a real number")
    tolerance = float(tie_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    if deformation.ndim != 4 or deformation.shape[-2:] != (3, 3) or not deformation.is_floating_point():
        raise ValueError("deformation must be a floating-point tensor of shape [B, C, 3, 3]")
    if not torch.isfinite(deformation).all():
        raise ValueError("deformation must contain only finite values")
    if reference is not None:
        if not isinstance(reference, Tensor):
            raise TypeError("reference must be a torch.Tensor or None")
        if (
            reference.shape != (deformation.shape[0], 3, 3)
            or reference.dtype != deformation.dtype
            or reference.device != deformation.device
        ):
            raise ValueError("reference must have shape [B, 3, 3] on the deformation dtype and device")
        if not torch.isfinite(reference).all():
            raise ValueError("reference must contain only finite values")
        reference = reference.detach()
        identity = torch.eye(3, dtype=reference.dtype, device=reference.device)
        orthogonality = (reference.transpose(-1, -2) @ reference - identity).abs().max()
        if orthogonality > 1e3 * torch.finfo(reference.dtype).eps or bool((torch.linalg.det(reference) <= 0).any()):
            raise ValueError("reference must contain proper rotations")

    with torch.no_grad():
        deformation = deformation.detach()
        left, singular_values, right_transpose = torch.linalg.svd(deformation)
        frames, orientation = _proper_rotation(left, right_transpose)
        scale = singular_values[..., 0].clamp_min(1.0)
        gap = torch.where(
            orientation > 0,
            singular_values[..., 1] + singular_values[..., 2],
            singular_values[..., 1] - singular_values[..., 2],
        )
        tie_mask = gap <= tolerance * scale
        if reference is not None and bool(tie_mask.any()):
            epsilon = (tolerance * scale)[..., None, None]
            tie_reference = reference[:, None].expand(-1, deformation.shape[1], -1, -1)[tie_mask].to(torch.float64)
            perturbed = deformation[tie_mask].to(torch.float64) + epsilon[tie_mask].to(torch.float64) * tie_reference
            tie_left, _, tie_right_transpose = torch.linalg.svd(perturbed)
            frames[tie_mask] = _proper_rotation(tie_left, tie_right_transpose)[0].to(deformation.dtype)
    return FrameResult(frames, tie_mask, singular_values)
