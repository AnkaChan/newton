# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental objective metric viscosity shared by hex energy and features."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["damping_metric_difference", "pack_damping_features"]


def damping_metric_difference(
    positions: torch.Tensor,
    previous_positions: torch.Tensor,
    cells: torch.Tensor,
    shape_gradients: torch.Tensor,
) -> torch.Tensor:
    """Return the dimensionless Gauss metric change with shape [B,N,8,3,3].

    Experimental. The caller validates geometry, topology, dtype and device.
    Both position operands retain their autograd history. The previous physical
    substep positions stay fixed throughout each optimization solve.

    Args:
        positions: Current shared corners [m], shape [B,P,3].
        previous_positions: Physical substep start [m], same shape/dtype/device.
        cells: Shared corner indices, shape [N,8].
        shape_gradients: Material shape gradients [1/m], shape [8,8,3].
    """
    import torch

    corners = positions[:, cells]
    previous_corners = previous_positions[:, cells]
    # Partition of unity permits subtracting one corner to reduce cancellation
    # under world translation without changing either material gradient.
    deformation = torch.einsum("bcki,qkj->bcqij", corners - corners[:, :, :1], shape_gradients)
    previous_deformation = torch.einsum(
        "bcki,qkj->bcqij", previous_corners - previous_corners[:, :, :1], shape_gradients
    )
    return deformation.transpose(-1, -2) @ deformation - previous_deformation.transpose(-1, -2) @ previous_deformation


def pack_damping_features(delta_C: torch.Tensor) -> torch.Tensor:
    """Pack dimensionless symmetric metrics into shape [B,N,48].

    Experimental. For each of eight Gauss points, emit xx, yy, zz, sqrt(2)xy,
    sqrt(2)xz, sqrt(2)yz. This packing preserves the Frobenius norm and remains
    invariant to world rigid motion. The caller supplies symmetric tensors.

    Args:
        delta_C: Gauss metric changes, shape [B,N,8,3,3].
    """
    import torch

    return torch.stack(
        (
            delta_C[..., 0, 0],
            delta_C[..., 1, 1],
            delta_C[..., 2, 2],
            2**0.5 * delta_C[..., 0, 1],
            2**0.5 * delta_C[..., 0, 2],
            2**0.5 * delta_C[..., 1, 2],
        ),
        dim=-1,
    ).flatten(-2)
