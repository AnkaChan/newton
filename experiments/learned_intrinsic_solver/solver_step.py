# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental learned update, differentiable hex fusion, and physical loss.

This proof of concept uses float32 by default. Construct with float64 only for
reference gradient checks. Torch work runs on the network's CPU or CUDA device.
The fixed PARDISO factorization stays on CPU with a custom forward/adjoint bridge;
construct a new step to change dtype, material, or constraints.

The single-material step consumes the same revised nine-value input schema as
the mixed-material step (:mod:`.features`, :mod:`.input_assembly`): cell frames
are the closest proper rotations to the current center deformation with the
clamped-face tie-break (:mod:`.frames`), recomputed per call without
differentiation; the gradient input is the detached position gradient of the
complete physical objective projected through the fusion adjoint; inverted and
collapsed candidates are accepted by the frames and the stable Neo-Hookean
energy. The network, other geometric features, corner reconstruction, and
energy remain connected.
"""

import torch  # noqa: TID253 -- Explicit opt-in PyTorch nn.Module implementation.
from torch import Tensor, nn  # noqa: TID253

from .data import VoxelGridData
from .features import (
    CONDITIONING_DIM,
    EDGE_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    STATE_FEATURE_DIM,
    center_deformation,
    conditioning_channels,
)
from .frames import select_reference_corners
from .fusion import HexFusion
from .hex_energy import HexImplicitEulerLoss
from .input_assembly import (
    LearnedHexInputs,
    LearnedHexStepOutput,
    OptimizerHistory,
    assemble_inputs,
    check_history,
)
from .network import IntrinsicSolverNetwork

__all__ = ["LearnedHexInputs", "LearnedHexSolverStep", "LearnedHexStepOutput"]


def _schema_error(network: IntrinsicSolverNetwork) -> str:
    schema = (network.state_feature_dim, network.conditioning_dim, network.edge_input_dim)
    return (
        f"network must use the revised schema {FEATURE_SCHEMA_VERSION}: {STATE_FEATURE_DIM} state, "
        f"{CONDITIONING_DIM} conditioning and {EDGE_FEATURE_DIM} edge inputs, got "
        f"{schema[0]}/{schema[1]}/{schema[2]}; legacy 38/5 and 86/6 networks and their checkpoints are not "
        "supported and require fresh initialization"
    )


class LearnedHexSolverStep(nn.Module):
    """Propose one learned optimization update for a clamped hexahedral body.

    Experimental: this is an optimizer iteration, not a committed physical time
    step or a converged simulation. Keep the physical inertial prediction and
    physical-start positions fixed across iterations within a time step.
    Inverted or collapsed candidates are accepted: each cell frame is the
    closest proper rotation to its center deformation, ties are broken with a
    reference frame from three prescribed corners of the clamped face (plain
    closest rotation when fewer than three noncollinear corners are
    prescribed), and the stable Neo-Hookean energy stays finite for every
    finite candidate. Only nonfinite or malformed inputs are rejected;
    acceptance/line search and contact remain outside this module. At least
    one corner must be prescribed.

    Inputs follow the revised schema shared with the mixed-material step
    through :func:`.input_assembly.assemble_inputs`:
    :data:`.features.STATE_FEATURE_DIM` state features (inertial axis offset,
    physical axis change, normalized current and previous axis gradients,
    normalized previous achieved update, exposed faces, fixed-corner flags,
    log gradient RMS and the history flag; packing in
    :func:`.features.pack_state_features`), the nine local axes prepended by
    the network, :data:`.features.EDGE_FEATURE_DIM` edge inputs and the six
    conditioning channels of :func:`.features.conditioning_channels`. The
    viscosity channel is always present; damping may be zero. Legacy 38/5 and
    86/6 networks and their checkpoints are rejected explicitly.

    Args:
        rest: Canonical full cuboid with cubic cells and z-fast corner ordering.
        fixed_indices: Indices of prescribed shared corners.
        lame_lambda: Nonnegative scalar or per-cell first Lamé parameter [Pa].
        lame_mu: Positive scalar or per-cell shear modulus [Pa].
        density: Scalar or per-cell rest density [kg/m^3].
        time_step: Positive physical time step [s].
        damping: Nonnegative scalar or per-cell metric viscosity [Pa*s].
        network: Optional revised-schema network on CPU or CUDA in the chosen
            dtype. Its device determines geometry, network, and energy
            execution. Default is the CPU one-block [1] baseline.
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
        damping=0.0,
        network: IntrinsicSolverNetwork | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.energy = HexImplicitEulerLoss(rest, lame_lambda, lame_mu, density, time_step, damping=damping, dtype=dtype)
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
        self.time_step = float(time_step)
        self.network = (
            network
            if network is not None
            else IntrinsicSolverNetwork(rest.cell_counts, STATE_FEATURE_DIM, conditioning_dim=CONDITIONING_DIM).to(
                dtype=dtype
            )
        )
        if self.network.cell_counts != rest.cell_counts:
            raise ValueError("network cell_counts must match the rest grid")
        schema = (self.network.state_feature_dim, self.network.conditioning_dim, self.network.edge_input_dim)
        if schema != (STATE_FEATURE_DIM, CONDITIONING_DIM, EDGE_FEATURE_DIM):
            raise ValueError(_schema_error(self.network))
        device = next(self.network.parameters()).device
        if device.type not in ("cpu", "cuda") or any(
            p.device != device or p.dtype != dtype for p in self.network.parameters()
        ):
            raise ValueError("network parameters must share a CPU or CUDA device and the constructor dtype")
        self.register_buffer("rest_positions", torch.as_tensor(rest.corner_rest_positions, dtype=dtype).clone())
        self.register_buffer("rest_centers", torch.as_tensor(rest.cell_rest_centers, dtype=dtype).clone())
        self.register_buffer("fixed_indices", torch.as_tensor(fixed_indices, dtype=torch.long).clone())
        reference_corners = select_reference_corners(rest.corner_rest_positions, self.fixed_indices)
        # Derived from rest geometry and prescribed IDs alone, so it is rebuilt at
        # construction and kept out of the state_dict. Empty when no three
        # noncollinear corners are pinned.
        self.register_buffer(
            "reference_corners",
            torch.empty(0, dtype=torch.long)
            if reference_corners is None
            else torch.as_tensor(reference_corners, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("cell_corner_indices", torch.as_tensor(rest.cell_corner_indices, dtype=torch.long).clone())
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=dtype)
        self.register_buffer("center_gradients", signs / (4 * rest.cell_size))
        flags = torch.zeros(len(rest.corner_rest_positions), dtype=dtype)
        flags[self.fixed_indices] = 1
        boundaries = torch.cat(
            (torch.as_tensor(rest.cell_exposed_faces, dtype=dtype), flags[self.cell_corner_indices]), -1
        )
        self.register_buffer("boundary_features", boundaries)
        # Per-cell material channels [C, 6]; the viscosity channel is present even at zero damping.
        self.register_buffer(
            "conditioning",
            conditioning_channels(lam, mu, rho, self.energy.damping, rest.cell_size, self.time_step),
        )
        self.to(device=device)

    def _check_positions(self, positions: Tensor, name: str) -> None:
        if not isinstance(positions, Tensor):
            raise ValueError(f"{name} must be a tensor")
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

    def _check_query(self, positions: Tensor, inertial_prediction: Tensor, previous_positions: Tensor | None) -> None:
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if inertial_prediction.shape != positions.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        if previous_positions is None:
            raise ValueError("previous_positions must supply the physical-step start positions")
        self._check_positions(previous_positions, "previous_positions")
        if previous_positions.shape != positions.shape:
            raise ValueError("positions and previous_positions must share a batch shape")

    def _energy_total(self, positions: Tensor, inertial_prediction: Tensor, previous_positions: Tensor) -> Tensor:
        return self.energy(positions, inertial_prediction, previous_positions=previous_positions).total

    def prepare_inputs(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        *,
        previous_positions: Tensor,
        frames: Tensor | None = None,
        history: OptimizerHistory | None = None,
    ) -> LearnedHexInputs:
        """Encode the candidate, physical Y, physical start and history into network inputs.

        Frames are the closest proper rotations to the center deformation F of
        each cell (:func:`.frames.closest_proper_rotations`); inverted cells
        get a right-handed frame whose local axes ``A = R^T F`` carry the
        negative determinant. Ambiguous cells use the reference frame built from
        ``reference_corners`` at the current positions when that buffer is
        nonempty. Frames and the gradient feature are detached so the
        decomposition is frozen for this query's backward pass. The state
        packing is documented in :func:`.input_assembly.assemble_inputs`.

        Args:
            positions: Candidate world corner positions [m], shape [B,P,3].
            inertial_prediction: Physical free-motion prediction [m], [B,P,3].
            previous_positions: Physical-step start [m], [B,P,3], held fixed
                across optimizer iterations. Required: it anchors the damping
                term and the physical axis-change block.
            frames: Optional precomputed rotations [B,C,3,3], validated as
                proper orthonormal rotations on the input dtype/device and
                detached; supply these to replay the same frozen frame.
            history: Detached previous-query history, or None for no history
                on the whole batch (zero blocks, ``history_valid = 0``).

        Returns:
            Frozen frames, differentiable local axes, packed features, the
            detached world axis gradient and zero-pinned position gradient, and
            the tie-break mask [B,C] (None when frames were supplied).

        Raises:
            ValueError: Malformed, mismatched or nonfinite inputs, missing
                previous positions, invalid supplied frames or history, or
                degenerate reference corners.
            TypeError: ``history`` is neither None nor an ``OptimizerHistory``.
        """
        self._check_query(positions, inertial_prediction, previous_positions)
        batch, cells = positions.shape[0], len(self.cell_corner_indices)
        history = check_history(history, batch=batch, cell_count=cells, dtype=positions.dtype, device=positions.device)
        if frames is not None:
            self._check_rotations(frames, (batch, cells, 3, 3), positions)
            frames = frames.detach()
        return assemble_inputs(
            self,
            positions,
            inertial_prediction,
            previous_positions,
            energy_total=self._energy_total,
            project_gradient=self.fusion.project_gradient,
            conditioning=self.conditioning[None].expand(batch, -1, -1),
            history=history,
            frames=frames,
        )

    def forward(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        *,
        previous_positions: Tensor,
        fixed_positions: Tensor | None = None,
        frames: Tensor | None = None,
        rigid_delta_rotation: Tensor | None = None,
        rigid_delta_translation: Tensor | None = None,
        detach_energy_target: bool = False,
        history: OptimizerHistory | None = None,
    ) -> LearnedHexStepOutput:
        """Update local axes, fuse shared corners, and evaluate implicit-Euler energy.

        Args:
            positions: Current candidate [m], shape [B,P,3].
            inertial_prediction: Unchanged physical Y [m], shape [B,P,3].
            previous_positions: Unchanged physical-step start [m], [B,P,3]. Required.
            fixed_positions: Prescribed positions [m], [B,K,3]; defaults to rest.
            frames: Optional frozen input frames [B,C,3,3], otherwise extracted.
            rigid_delta_rotation: Optional world rotation [B,3,3] carrying the
                current pose to the proposed pose; used only for fusion.
            rigid_delta_translation: Optional world translation [m], [B,3], in
                the same map x -> Q*x+t. Pins set the final translation in this
                clamped baseline, so this t cancels from the exact minimizer.
            detach_energy_target: Treat the inertial predictor and physical-start
                positions as fixed only in the physical energy. Their network
                feature paths remain differentiable, including in consecutive
                physical steps.
            history: Detached previous-query history or None.

        Returns:
            Proposed global positions and per-object physical energy terms.
            ``step_size`` is the per-cell step [B,C]. The detached diagnostics
            are filled: ``axis_gradient_world`` (this query's world axis
            gradient feature [J]), ``achieved_axis_update_world`` (world change
            of the center deformation from ``positions`` to the fused output,
            including any rigid delta applied in fusion), ``force_residual_norm``
            (norm of the zero-pinned position gradient at the pre-update
            candidate [N]) and ``tie_mask``. No state is mutated or physical
            time advanced.
        """
        if not isinstance(detach_energy_target, bool):
            raise TypeError("detach_energy_target must be boolean")
        inputs = self.prepare_inputs(
            positions, inertial_prediction, previous_positions=previous_positions, frames=frames, history=history
        )
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
        energy_previous = previous_positions.detach() if detach_energy_target else previous_positions
        loss = self.energy(fused, energy_target, previous_positions=energy_previous)
        with torch.no_grad():
            achieved = center_deformation(
                fused.detach(), self.cell_corner_indices, self.center_gradients
            ) - center_deformation(positions.detach(), self.cell_corner_indices, self.center_gradients)
            residual = torch.linalg.vector_norm(inputs.position_gradient.flatten(1), dim=1)
        return LearnedHexStepOutput(
            fused,
            prediction.local_target_axes,
            prediction.axis_correction,
            prediction.step_size,
            inputs.frames,
            loss,
            axis_gradient_world=inputs.axis_gradient_world,
            achieved_axis_update_world=achieved,
            force_residual_norm=residual,
            tie_mask=inputs.tie_mask,
        )
