# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental cuboid neighborhoods and invariant cell-frame edge features.

This module supports fully occupied canonical cuboids only. Neighborhoods do
not represent holes, disconnected occupancy, or arbitrary material graphs.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["build_edge_features", "build_grid_neighborhood"]


def build_grid_neighborhood(
    cell_counts: tuple[int, int, int],
    hop: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a 26-connected exact-hop shell plus self for an occupied cuboid.

    Experimental: this topology assumes all cells are occupied. Cell IDs use
    z-fast ordering. Slot zero contains self; remaining slots enumerate offsets
    lexicographically at exactly the requested Chebyshev distance. One hop
    includes cells sharing a face, edge, or corner. Hops 1, 2, and 4 have 27,
    99, and 387 slots, including masked boundary slots.

    Args:
        cell_counts: Positive integer cell counts along material x, y, and z.
        hop: Positive integer topological distance of the neighbor shell.
        device: Device on which to create the topology tensors.

    Returns:
        Long neighbor IDs and boolean validity, both shaped [N, S]. Invalid
        slots have ID zero and false validity. Self is valid for every cell.

    Raises:
        ValueError: If the counts or hop are not positive integers.
    """
    import torch

    if (
        not isinstance(cell_counts, tuple)
        or len(cell_counts) != 3
        or any(isinstance(count, bool) or not isinstance(count, Integral) or count <= 0 for count in cell_counts)
    ):
        raise ValueError("cell_counts must be a tuple of three positive integers")
    if isinstance(hop, bool) or not isinstance(hop, Integral) or hop <= 0:
        raise ValueError("hop must be a positive integer")

    nx, ny, nz = (int(count) for count in cell_counts)
    hop = int(hop)
    offsets = [(0, 0, 0)] + [
        (dx, dy, dz)
        for dx in range(-hop, hop + 1)
        for dy in range(-hop, hop + 1)
        for dz in range(-hop, hop + 1)
        if max(abs(dx), abs(dy), abs(dz)) == hop
    ]

    cell_ids = torch.arange(nx * ny * nz, dtype=torch.long, device=device)
    coordinates = torch.stack((cell_ids // (ny * nz), cell_ids // nz % ny, cell_ids % nz), dim=-1)
    neighbor_coordinates = coordinates[:, None, :] + torch.tensor(offsets, dtype=torch.long, device=device)
    counts = torch.tensor((nx, ny, nz), dtype=torch.long, device=device)
    valid = (neighbor_coordinates >= 0).all(dim=-1) & (neighbor_coordinates < counts).all(dim=-1)
    indices = (
        neighbor_coordinates[..., 0] * (ny * nz) + neighbor_coordinates[..., 1] * nz + neighbor_coordinates[..., 2]
    )
    return indices.masked_fill(~valid, 0), valid


def build_edge_features(
    rest_centers: torch.Tensor,
    current_centers: torch.Tensor,
    frames: torch.Tensor,
    local_axes: torch.Tensor,
    cell_size: float,
    neighbor_indices: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """Express neighbor geometry in the receiver's detached current frame.

    Experimental: all geometry tensors must share one floating dtype and
    device. Float32 inputs produce float32 features without promotion. Frames
    are treated as rotations and detached here; other input gradient paths
    remain available. Material reference axes are common across cells.

    Feature order is rest offset / h (3), current receiver-frame offset / h
    (3), relative frame (9), and transported neighbor axes (9). Matrices store
    axes in columns and flatten in row-major order. All features are
    dimensionless. A valid self edge retains its identity relative frame and
    own axes; it is not an all-zero token.

    Args:
        rest_centers: Material rest positions [m], shape [N, 3].
        current_centers: World positions [m], shape [B, N, 3].
        frames: Local-to-world rotations, shape [B, N, 3, 3].
        local_axes: Deformation axes in each cell's frame, shape [B, N, 3, 3].
        cell_size: Positive finite rest voxel edge length [m].
        neighbor_indices: Long neighbor IDs shared across batches, shape [N, S].
            IDs at valid slots must lie in [0, N). Masked slots may use any ID.
        neighbor_mask: Boolean neighbor validity, shape [N, S].

    Returns:
        Features shaped [B, N, S, 24], with every masked slot set to zero.

    Raises:
        TypeError: If tensor dtypes are incompatible or an input is not a tensor.
        ValueError: If shapes, devices, or cell size are incompatible.
    """
    import torch

    tensors = (
        ("rest_centers", rest_centers),
        ("current_centers", current_centers),
        ("frames", frames),
        ("local_axes", local_axes),
        ("neighbor_indices", neighbor_indices),
        ("neighbor_mask", neighbor_mask),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if rest_centers.ndim != 2 or rest_centers.shape[-1] != 3 or rest_centers.shape[0] == 0:
        raise ValueError("rest_centers must have shape [N, 3] with N > 0")
    cell_count = rest_centers.shape[0]
    if current_centers.ndim != 3 or current_centers.shape[1:] != (cell_count, 3):
        raise ValueError("current_centers must have shape [B, N, 3]")
    batch_count = current_centers.shape[0]
    for name, tensor in (("frames", frames), ("local_axes", local_axes)):
        if tensor.shape != (batch_count, cell_count, 3, 3):
            raise ValueError(f"{name} must have shape [B, N, 3, 3]")
    if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != cell_count:
        raise ValueError("neighbor_indices must have shape [N, S]")
    if neighbor_mask.shape != neighbor_indices.shape:
        raise ValueError("neighbor_mask must have the same [N, S] shape as neighbor_indices")

    for name, tensor in tensors[:4]:
        if not tensor.is_floating_point() or tensor.dtype != rest_centers.dtype:
            raise TypeError(f"{name} must share the floating dtype of rest_centers")
    if neighbor_indices.dtype != torch.long:
        raise TypeError("neighbor_indices must have dtype torch.long")
    if neighbor_mask.dtype != torch.bool:
        raise TypeError("neighbor_mask must have dtype torch.bool")
    for name, tensor in tensors[1:]:
        if tensor.device != rest_centers.device:
            raise ValueError(f"{name} must be on the same device as rest_centers")
    if isinstance(cell_size, bool) or not isinstance(cell_size, Real) or not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be a positive finite number")

    # Sanitize padding before gathering: sentinel IDs need not be in range.
    safe_indices = neighbor_indices.masked_fill(~neighbor_mask, 0)
    rotations = frames.detach()
    receiver_transpose = rotations.transpose(-1, -2).unsqueeze(2)
    relative_frames = receiver_transpose @ rotations[:, safe_indices]
    relative_axes = relative_frames @ local_axes[:, safe_indices]
    current_offsets = current_centers[:, safe_indices] - current_centers.unsqueeze(2)
    local_offsets = (receiver_transpose @ current_offsets.unsqueeze(-1)).squeeze(-1) / cell_size
    rest_offsets = (rest_centers[safe_indices] - rest_centers.unsqueeze(1)) / cell_size
    features = torch.cat(
        (
            rest_offsets.unsqueeze(0).expand(batch_count, -1, -1, -1),
            local_offsets,
            relative_frames.flatten(start_dim=-2),
            relative_axes.flatten(start_dim=-2),
        ),
        dim=-1,
    )
    return features.masked_fill(~neighbor_mask[None, :, :, None], 0)
