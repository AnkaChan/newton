# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental orientation checks and bounded shortening of learned proposals."""

import torch  # noqa: TID253 -- Optional experimental Torch geometry.
from torch import Tensor, nn  # noqa: TID253

from .data import VoxelGridData
from .multiscale import _GeometryScreen

__all__ = ["HexFeasibility"]


class HexFeasibility(nn.Module):
    """Shorten experimental learned updates until sampled orientation is valid.

    Use the augmenter's eight corners, eight Gauss points, cell centers and
    alternating five-tet topology. The center also satisfies the solver's
    float32 singular-value threshold. Determinants must clear a scale-aware
    roundoff margin, including cancellation in the hex gradient accumulation.
    These samples do not certify continuous
    injectivity or exclude self-contact. All runtime calculations stay float32
    on the input device; the shared NumPy construction runs only at setup.

    The selected scale is detached. Gradients pass through the accepted linear
    interpolation, including its proposal/fusion graph. A valid full proposal
    remains bit exact. Nonfinite proposals, invalid bases and exhaustion of
    32 halvings raise explicitly rather than silently freezing a trajectory.

    Args:
        rest: Canonical cubic hexahedral grid in meters.
    """

    def __init__(self, rest: VoxelGridData):
        super().__init__()
        screen = _GeometryScreen(rest)
        self.particle_count = len(rest.corner_rest_positions)
        self.register_buffer("cells", torch.tensor(rest.cell_corner_indices, dtype=torch.long), persistent=False)
        self.register_buffer("gradients", torch.tensor(screen.derivatives, dtype=torch.float32), persistent=False)
        self.register_buffer("tets", torch.tensor(screen.tets, dtype=torch.long), persistent=False)
        self.register_buffer(
            "rest_tet_determinants", torch.tensor(screen.rest_tet_determinants, dtype=torch.float32), persistent=False
        )

    def _check_shape(self, positions: Tensor) -> None:
        if positions.ndim != 3 or positions.shape[1:] != (self.particle_count, 3) or positions.shape[0] == 0:
            raise ValueError("positions must have shape [nonempty_batch, corner_count, 3]")
        if positions.dtype != torch.float32 or positions.device != self.gradients.device:
            raise ValueError("positions must use float32 on the geometry device")

    @torch.no_grad()
    def valid(self, positions: Tensor) -> Tensor:
        """Return detached per-instance sampled orientation flags, shape [B]."""
        self._check_shape(positions)
        corners = positions[:, self.cells]
        relative = corners - corners[:, :, :1]
        deformation = torch.einsum("bcki,qkj->bcqij", relative, self.gradients)
        absolute_accumulation = torch.einsum("bcki,qkj->bcqij", relative.abs(), self.gradients.abs())
        # Cover the eight-corner sums and the determinant's three columns;
        # actual F norms alone miss cancellation during gradient accumulation.
        roundoff = 64 * torch.finfo(positions.dtype).eps
        jacobian_margin = roundoff * torch.linalg.vector_norm(absolute_accumulation, dim=-2).prod(-1)
        jacobian = torch.linalg.det(deformation)
        finite = torch.isfinite(deformation).all((-1, -2, -3))
        center = deformation[:, :, -1]
        safe_center = torch.where(
            finite[..., None, None], center, torch.eye(3, dtype=positions.dtype, device=positions.device)
        )
        singular = torch.linalg.svdvals(safe_center)
        threshold = 4 * torch.finfo(positions.dtype).eps * singular[..., 0].clamp_min(1)
        tetrahedra = positions[:, self.tets]
        tet_edges = (tetrahedra[:, :, :, 1:] - tetrahedra[:, :, :, :1]).transpose(-1, -2)
        tet_determinant = torch.linalg.det(tet_edges)
        tet_margin = roundoff * torch.linalg.vector_norm(tet_edges, dim=-2).prod(-1)
        tet_ratio = tet_determinant / self.rest_tet_determinants
        return (
            torch.isfinite(positions).all((-1, -2))
            & finite.all(-1)
            & torch.isfinite(jacobian).all((-1, -2))
            & (jacobian > jacobian_margin).all((-1, -2))
            & (singular[..., -1] > threshold).all(-1)
            & torch.isfinite(tet_ratio).all((-1, -2))
            & (tet_ratio > tet_margin / self.rest_tet_determinants.abs()).all((-1, -2))
        )

    def forward(self, base: Tensor, proposal: Tensor) -> tuple[Tensor, Tensor]:
        """Return feasible positions [B,P,3] and detached accepted scales [B]."""
        self._check_shape(base)
        self._check_shape(proposal)
        if base.shape != proposal.shape:
            raise ValueError("base and proposal must have matching batches")
        with torch.no_grad():
            if not torch.isfinite(base).all() or not torch.isfinite(proposal).all():
                raise ValueError("base and proposal positions must be finite")
            if not self.valid(base).all():
                raise ValueError("base shape fails sampled hex orientation or center nonsingularity")
            scale = proposal.new_ones(len(proposal))
            accepted = self.valid(proposal)
            for _ in range(32):
                if accepted.all():
                    break
                scale = torch.where(accepted, scale, scale * 0.5)
                trial = torch.where(
                    (scale == 1)[:, None, None], proposal, base + scale[:, None, None] * (proposal - base)
                )
                accepted = self.valid(trial)
            if not accepted.all():
                failed = torch.nonzero(~accepted).flatten().tolist()
                raise ValueError(f"hex feasibility failed after 32 halvings for batch members {failed}")
        positions = torch.where((scale == 1)[:, None, None], proposal, base + scale[:, None, None] * (proposal - base))
        return positions, scale
