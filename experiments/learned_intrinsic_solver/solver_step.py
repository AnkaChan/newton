# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental learned update, differentiable hex fusion, and physical loss.

This proof of concept uses float32 by default. Construct with float64 only for
reference gradient checks. Torch work runs on the network's CPU or CUDA device.
The fixed PARDISO factorization stays on CPU with a custom forward/adjoint bridge;
construct a new step to change dtype, material, or constraints.
Polar rotations are recomputed per call without differentiation. The network,
other geometric features, corner reconstruction, and energy remain connected.
"""

from typing import NamedTuple

import torch  # noqa: TID253 -- Explicit opt-in PyTorch nn.Module implementation.
from torch import Tensor, nn  # noqa: TID253

from .data import VoxelGridData
from .fusion import HexFusion
from .hex_energy import HexImplicitEulerLoss, HexLossTerms
from .network import IntrinsicSolverNetwork
from .network_geometry import build_edge_features

__all__ = ["LearnedHexInputs", "LearnedHexSolverStep", "LearnedHexStepOutput"]


class LearnedHexInputs(NamedTuple):
    """Experimental packed geometry and features for one network evaluation."""

    frames: Tensor
    local_axes: Tensor
    state_features: Tensor
    edge_features: dict[int, Tensor]
    conditioning: Tensor


class LearnedHexStepOutput(NamedTuple):
    """Experimental shared corners [m], local targets, frozen frames, and energy [J]."""

    positions: Tensor
    local_target_axes: Tensor
    axis_correction: Tensor
    step_size: Tensor
    frames: Tensor
    loss: HexLossTerms


class LearnedHexSolverStep(nn.Module):
    """Propose one learned optimization update for a clamped hexahedral body.

    Experimental: this is an optimizer iteration, not a committed physical time
    step or a converged simulation. Keep the physical inertial prediction fixed
    across iterations within a time step. A candidate with a nonpositive Gauss
    Jacobian is rejected by the energy; acceptance/line search and contact remain
    outside this module. At least one corner must be prescribed.

    The baseline packs 38 state features: eight receiver-frame inertial vectors
    divided by rest edge length (24), exposed faces (6), and fixed corners (8).
    The network adds the nine local axes. Five FiLM channels are
    log1p(lambda/1e5 Pa), log1p(mu/1e5 Pa), log(rho/1000 kg/m^3),
    log(h/0.025 m), log(dt/(1/60 s)).

    Args:
        rest: Canonical full cuboid with cubic cells and z-fast corner ordering.
        fixed_indices: Indices of prescribed shared corners.
        lame_lambda: Nonnegative scalar or per-cell first Lamé parameter [Pa].
        lame_mu: Positive scalar or per-cell shear modulus [Pa].
        density: Scalar or per-cell rest density [kg/m^3].
        time_step: Positive physical time step [s].
        network: Optional network with 38 state, 24 edge, and 5 conditioning
            inputs on CPU or CUDA in the chosen dtype. Its device determines
            geometry, network, and energy execution. Default is the CPU [1,1,1] baseline.
        dtype: Working Torch dtype; float32 default, float64 reference only.
    """

    def __init__(
        self,
        rest: VoxelGridData,
        fixed_indices,
        *,
        lame_lambda,
        lame_mu,
        density,
        time_step: float,
        network: IntrinsicSolverNetwork | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.energy = HexImplicitEulerLoss(rest, lame_lambda, lame_mu, density, time_step, dtype=dtype)
        lam = self.energy.lame_lambda
        mu = self.energy.lame_mu
        rho = self.energy.density
        # Preserve the former E*V fusion weights using the equivalent Young
        # stiffness. The constitutive law and network use Lamé inputs directly.
        material_scale = torch.maximum(lam, mu)
        mu_fraction = (mu / material_scale) / (lam / material_scale + mu / material_scale)
        fusion_stiffness = mu * (3 - mu_fraction)
        self.register_buffer("fusion_stiffness", fusion_stiffness)
        self.fusion = HexFusion(rest, fixed_indices, cell_weights=fusion_stiffness * rest.cell_size**3, dtype=dtype)
        self.cell_size = rest.cell_size
        self.network = network if network is not None else IntrinsicSolverNetwork(rest.cell_counts, 38).to(dtype=dtype)
        if (
            self.network.cell_counts != rest.cell_counts
            or self.network.state_feature_dim != 38
            or self.network.conditioning_dim != 5
            or self.network.edge_input_dim != 24
        ):
            raise ValueError("network must match the grid and have 38 state, 5 conditioning, and 24 edge inputs")
        device = next(self.network.parameters()).device
        if device.type not in ("cpu", "cuda") or any(
            p.device != device or p.dtype != dtype for p in self.network.parameters()
        ):
            raise ValueError("network parameters must share a CPU or CUDA device and the constructor dtype")
        self.register_buffer("rest_positions", torch.as_tensor(rest.corner_rest_positions, dtype=dtype).clone())
        self.register_buffer("rest_centers", torch.as_tensor(rest.cell_rest_centers, dtype=dtype).clone())
        self.register_buffer("fixed_indices", torch.as_tensor(fixed_indices, dtype=torch.long).clone())
        self.register_buffer("cell_corner_indices", torch.as_tensor(rest.cell_corner_indices, dtype=torch.long).clone())
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=dtype)
        self.register_buffer("center_gradients", signs / (4 * rest.cell_size))
        flags = torch.zeros(len(rest.corner_rest_positions), dtype=dtype)
        flags[self.fixed_indices] = 1
        boundaries = torch.cat(
            (torch.as_tensor(rest.cell_exposed_faces, dtype=dtype), flags[self.cell_corner_indices]), -1
        )
        self.register_buffer("boundary_features", boundaries)
        conditioning = torch.stack(
            (
                (lam / 1e5).log1p(),
                (mu / 1e5).log1p(),
                (rho / 1000).log(),
                torch.full_like(mu, rest.cell_size / 0.025).log(),
                torch.full_like(mu, float(time_step) * 60).log(),
            ),
            dim=-1,
        )
        self.register_buffer("conditioning", conditioning)
        self.to(device=device)

    def _check_positions(self, positions: Tensor, name: str) -> None:
        if positions.ndim != 3 or positions.shape[1:] != self.rest_positions.shape or not positions.shape[0]:
            raise ValueError(f"{name} must have shape [nonempty_batch, corner_count, 3]")
        if positions.device != self.rest_positions.device or positions.dtype != self.rest_positions.dtype:
            raise ValueError(f"{name} must match the module device and constructor dtype")
        if not torch.isfinite(positions).all():
            raise ValueError(f"{name} must be finite")

    @staticmethod
    def _check_rotations(rotations: Tensor, shape: tuple, reference: Tensor) -> None:
        if rotations.shape != shape or rotations.dtype != reference.dtype or rotations.device != reference.device:
            raise ValueError("rotations must match the expected shape, dtype, and device")
        tolerance = 5e-5 if reference.dtype == torch.float32 else 1e-10
        eye = torch.eye(3, dtype=reference.dtype, device=reference.device).expand(shape)
        if not torch.allclose(rotations.transpose(-1, -2) @ rotations, eye, atol=tolerance, rtol=tolerance):
            raise ValueError("frames and rigid rotations must be orthonormal")
        if not (torch.linalg.det(rotations) > 0).all():
            raise ValueError("frames and rigid rotations must have positive determinant")

    def prepare_inputs(
        self, positions: Tensor, inertial_prediction: Tensor, *, frames: Tensor | None = None
    ) -> LearnedHexInputs:
        """Encode current corners and unchanged physical Y into network inputs.

        Args:
            positions: Candidate world corner positions [m], shape [B,P,3].
            inertial_prediction: Physical free-motion prediction [m], [B,P,3].
            frames: Optional precomputed rotations [B,C,3,3]. They are detached;
                supply these to hold the same frame during geometry checks.

        Returns:
            Frozen polar frames, differentiable local axes, and packed features.
        """
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if inertial_prediction.shape != positions.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        corners = positions[:, self.cell_corner_indices]
        deformation = torch.einsum("bcki,kj->bcij", corners - corners[:, :, :1], self.center_gradients)
        if frames is None:
            with torch.no_grad():
                left, singular, right_transpose = torch.linalg.svd(deformation)
                threshold = 4 * torch.finfo(positions.dtype).eps * singular[..., 0].clamp_min(1)
                if (torch.linalg.det(deformation) <= 0).any() or (singular[..., -1] <= threshold).any():
                    raise ValueError("current center deformation must be positively oriented and nonsingular")
                frames = left @ right_transpose
        else:
            self._check_rotations(frames, deformation.shape, positions)
            frames = frames.detach()
        axes = frames.transpose(-1, -2) @ deformation
        offsets = inertial_prediction[:, self.cell_corner_indices] - corners
        local_offsets = torch.einsum("bcij,bckj->bcki", frames.transpose(-1, -2), offsets) / self.cell_size
        batch = positions.shape[0]
        state = torch.cat((local_offsets.flatten(-2), self.boundary_features[None].expand(batch, -1, -1)), dim=-1)
        edges = {
            hop: build_edge_features(
                self.rest_centers, corners.mean(-2), frames, axes, self.cell_size, *self.network.neighborhood(hop)
            )
            for hop in set(self.network.hops)
        }
        return LearnedHexInputs(frames, axes, state, edges, self.conditioning[None].expand(batch, -1, -1))

    def forward(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        *,
        fixed_positions: Tensor | None = None,
        frames: Tensor | None = None,
        rigid_delta_rotation: Tensor | None = None,
        rigid_delta_translation: Tensor | None = None,
        detach_energy_target: bool = False,
    ) -> LearnedHexStepOutput:
        """Update local axes, fuse shared corners, and evaluate implicit-Euler energy.

        Args:
            positions: Current candidate [m], shape [B,P,3].
            inertial_prediction: Unchanged physical Y [m], shape [B,P,3].
            fixed_positions: Prescribed positions [m], [B,K,3]; defaults to rest.
            frames: Optional frozen input frames [B,C,3,3], otherwise extracted.
            rigid_delta_rotation: Optional world rotation [B,3,3] carrying the
                current pose to the proposed pose; used only for fusion.
            rigid_delta_translation: Optional world translation [m], [B,3], in
                the same map x -> Q*x+t. Pins set the final translation in this
                clamped baseline, so this t cancels from the exact minimizer.
            detach_energy_target: Treat the inertial predictor as fixed only
                in the physical energy. Its network-feature path remains
                differentiable, including in consecutive physical steps.

        Returns:
            Proposed global positions and per-object physical energy terms.
            No state is mutated or physical time advanced.
        """
        if not isinstance(detach_energy_target, bool):
            raise TypeError("detach_energy_target must be boolean")
        inputs = self.prepare_inputs(positions, inertial_prediction, frames=frames)
        prediction = self.network(inputs.local_axes, inputs.state_features, inputs.edge_features, inputs.conditioning)
        world_increment = inputs.frames @ (prediction.local_target_axes - inputs.local_axes)
        base = positions
        batch = positions.shape[0]
        if rigid_delta_rotation is not None:
            self._check_rotations(rigid_delta_rotation, (batch, 3, 3), positions)
            base = positions @ rigid_delta_rotation.transpose(-1, -2)
            world_increment = rigid_delta_rotation[:, None] @ world_increment
        if rigid_delta_translation is not None:
            if (
                rigid_delta_translation.shape != (batch, 3)
                or rigid_delta_translation.dtype != positions.dtype
                or rigid_delta_translation.device != positions.device
                or not torch.isfinite(rigid_delta_translation).all()
            ):
                raise ValueError("rigid_delta_translation must be finite [B,3] on the input dtype/device")
            base = base + rigid_delta_translation[:, None]
        if fixed_positions is None:
            fixed_positions = self.rest_positions[self.fixed_indices][None].expand(batch, -1, -1)
        fused = self.fusion.fuse(base, world_increment, fixed_positions)
        energy_target = inertial_prediction.detach() if detach_energy_target else inertial_prediction
        loss = self.energy(fused, energy_target)
        return LearnedHexStepOutput(
            fused, prediction.local_target_axes, prediction.axis_correction, prediction.step_size, inputs.frames, loss
        )
