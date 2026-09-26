# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental heterogeneous physical contexts sharing one batched network.

Geometry, the revised nine-value input schema, and eight-point hex energy use
batched float32 Torch. Each material owns a CPU PARDISO factor and a native
Newton rigid predictor. Contexts are ordinary Python objects, excluded from
module state and DDP broadcasts.

Frames are the closest proper rotations of the cell-center deformation with the
clamped-face tie-break from :mod:`frames`. Inverted and collapsed candidates
are accepted: the Newton stable Neo-Hookean law is finite for every finite
shape, no geometry backtracking or acceptance scaling exists, and only
nonfinite inputs raise. The gradient input is the detached position gradient of
the complete physical objective projected through each context's fusion
adjoint, normalized with the LeCO convention in :mod:`features`.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor, nn  # noqa: TID253

from .damping import damping_metric_difference
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
from .hex_energy import HexImplicitEulerLoss, HexLossTerms, make_inertial_prediction, stable_neo_hookean_density
from .input_assembly import (
    LearnedHexInputs,
    LearnedHexStepOutput,
    OptimizerHistory,
    assemble_inputs,
    check_history,
)
from .network import IntrinsicSolverNetwork
from .rigid_predictor import RigidPosePredictor

__all__ = ["MixedHexSolverStep", "OptimizerHistory"]

_FLOAT32_EPSILON = 2.0**-23
"""Unit roundoff of float32 used by the material-aware energy floor."""


@dataclass
class _PhysicalContext:
    specification: dict[str, float]
    material: Tensor
    mass: Tensor
    fusion: HexFusion
    predictor: RigidPosePredictor
    lock: threading.RLock = field(default_factory=threading.RLock)


class MixedHexSolverStep(nn.Module):
    """Apply one learned update to objects with distinct homogeneous materials.

    Experimental. All objects share a canonical grid, pin indices, gravity and
    timestep. Each batch evaluates the shared network exactly once. Register
    material contexts on CPU before using their IDs; registration and physical
    preparation never read trainable weights. Contexts may be created by worker
    threads while another batch runs. Wrap this module with DDP using
    ``broadcast_buffers=False`` and checkpoint ``context_specs`` separately.

    Only the revised schema is supported: :data:`features.STATE_FEATURE_DIM`
    state inputs (five nine-value matrix blocks in the receiving frame,
    boundary flags, log gradient RMS and the history flag), six conditioning
    channels and 24 edge inputs. Frames are the closest proper rotations of the
    cell-center deformation; ties are broken with the reference built from
    three prescribed corners of the clamped face when
    :func:`frames.select_reference_corners` finds them, otherwise the plain
    formula is kept. The frame decomposition and the gradient feature are
    frozen for the query; the network, local axes, fusion and energy remain
    differentiable to every network parameter.

    The rigid predictor only initializes the candidate. The physical inertial
    target stays unchanged through learned updates. Inverted or collapsed
    candidates are evaluated with the stable Neo-Hookean law and are never
    rejected or shortened; nonfinite inputs raise. Contact and energy-descent
    acceptance are not implemented.

    Args:
        rest: Canonical cubic hexahedral rest grid [m].
        fixed_indices: Unique prescribed corner indices, at least one.
        network: Shared float32 network with the revised schema, on CPU or CUDA.
        time_step: Positive physical timestep [s].
        gravity: World acceleration [m/s^2].
        energy_floor_scale: Positive multiplier ``c`` of the material-aware
            energy floor returned by :meth:`energy_floor`.
    """

    def __init__(
        self,
        rest: VoxelGridData,
        fixed_indices,
        *,
        network: IntrinsicSolverNetwork,
        time_step: float,
        gravity=(0.0, -9.81, 0.0),
        energy_floor_scale: float = 1.0,
    ):
        super().__init__()
        if (
            isinstance(time_step, bool)
            or not isinstance(time_step, Real)
            or not math.isfinite(time_step)
            or time_step <= 0
        ):
            raise ValueError("time_step must be finite and positive")
        if (
            isinstance(energy_floor_scale, bool)
            or not isinstance(energy_floor_scale, Real)
            or not math.isfinite(energy_floor_scale)
            or energy_floor_scale <= 0
        ):
            raise ValueError("energy_floor_scale must be finite and positive")
        if network.cell_counts != rest.cell_counts:
            raise ValueError("network cell_counts must match the rest grid")
        schema = (network.state_feature_dim, network.conditioning_dim, network.edge_input_dim)
        if schema != (STATE_FEATURE_DIM, CONDITIONING_DIM, EDGE_FEATURE_DIM):
            raise ValueError(
                f"network must use the revised schema {FEATURE_SCHEMA_VERSION}: {STATE_FEATURE_DIM} state, "
                f"{CONDITIONING_DIM} conditioning and {EDGE_FEATURE_DIM} edge inputs, got "
                f"{schema[0]}/{schema[1]}/{schema[2]}; legacy 38/5 and 86/6 networks are not supported"
            )
        device = next(network.parameters()).device
        if device.type not in ("cpu", "cuda") or any(
            parameter.device != device or parameter.dtype != torch.float32 for parameter in network.parameters()
        ):
            raise ValueError("network parameters must share a CPU or CUDA device and float32 dtype")
        if isinstance(fixed_indices, Tensor):
            if fixed_indices.device.type != "cpu":
                raise ValueError("fixed_indices must be on CPU")
            fixed_indices = fixed_indices.detach().numpy()
        fixed = np.asarray(fixed_indices)
        count = len(rest.corner_rest_positions)
        if (
            fixed.ndim != 1
            or not fixed.size
            or fixed.dtype.kind not in "iu"
            or np.any(fixed < 0)
            or np.any(fixed >= count)
            or len(np.unique(fixed)) != len(fixed)
        ):
            raise ValueError("fixed_indices must contain unique in-range corner indices and at least one pin")
        gravity_tensor = torch.as_tensor(gravity, dtype=torch.float32, device="cpu").detach().clone()
        if gravity_tensor.shape != (3,) or not torch.isfinite(gravity_tensor).all():
            raise ValueError("gravity must be a finite three-vector")
        self._rest = rest
        self._fixed_cpu = torch.tensor(fixed.copy(), dtype=torch.long)
        self._gravity_cpu = gravity_tensor
        self.time_step = float(time_step)
        self.cell_size = rest.cell_size
        self.energy_floor_scale = float(energy_floor_scale)
        self.rest_volume = len(rest.cell_corner_indices) * rest.cell_size**3
        self.network = network
        self._contexts: dict[str, _PhysicalContext] = {}
        self._contexts_lock = threading.RLock()
        self._build_lock = threading.Lock()
        # Reuse existing canonical-grid and quadrature validation without a
        # network call or a material-dependent module attached to this module.
        geometry = HexImplicitEulerLoss(rest, 0.0, 1.0, 1.0, self.time_step)
        self.register_buffer("cell_corner_indices", geometry.cell_corner_indices)
        self.register_buffer("shape_gradients", geometry.shape_gradients)
        self.register_buffer("quadrature_weights", geometry.quadrature_weights)
        self.register_buffer("rest_positions", torch.tensor(rest.corner_rest_positions, dtype=torch.float32))
        self.register_buffer("rest_centers", torch.tensor(rest.cell_rest_centers, dtype=torch.float32))
        self.register_buffer("fixed_indices", self._fixed_cpu.clone())
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
        self.register_buffer("center_gradients", signs / (4 * rest.cell_size))
        flags = torch.zeros(count, dtype=torch.float32)
        flags[self.fixed_indices] = 1
        boundary = torch.cat(
            (torch.tensor(rest.cell_exposed_faces, dtype=torch.float32), flags[self.cell_corner_indices]), -1
        )
        self.register_buffer("boundary_features", boundary)
        corners = select_reference_corners(rest.corner_rest_positions, fixed)
        reference = torch.zeros(0, dtype=torch.long) if corners is None else torch.as_tensor(corners, dtype=torch.long)
        self.register_buffer("reference_corners", reference)
        self.to(device=device)

    @property
    def context_specs(self) -> dict[str, dict[str, float]]:
        """Return independent material [Pa, kg/m^3] and damping [Pa*s] specifications."""
        with self._contexts_lock:
            return {name: context.specification.copy() for name, context in self._contexts.items()}

    def register_context(
        self, context_id: str, *, lame_lambda: float, lame_mu: float, density: float, damping: float = 0.0
    ) -> None:
        """Build one CPU material/factor/predictor context without consulting weights.

        Args:
            context_id: Unique nonempty identifier for replay payloads.
            lame_lambda: Nonnegative first Lamé parameter [Pa].
            lame_mu: Positive shear modulus [Pa].
            density: Positive rest density [kg/m^3].
            damping: Nonnegative absolute damping coefficient [Pa*s], held
                fixed for this context.
        """
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("context_id must be a nonempty string")
        specification = {"lame_lambda": lame_lambda, "lame_mu": lame_mu, "density": density, "damping": damping}
        for name, value in specification.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            if value < 0 or (name not in ("lame_lambda", "damping") and value == 0):
                raise ValueError(f"{name} is outside the physical range")
        specification = {name: float(value) for name, value in specification.items()}
        damping_tensor = torch.tensor(damping, dtype=torch.float32)
        if not torch.isfinite(damping_tensor):
            raise ValueError("damping must remain finite in float32")
        # Serialize native construction independently of forward lookups. Warp
        # setup and factor allocation are never performed under the registry lock.
        with self._build_lock:
            with self._contexts_lock:
                if context_id in self._contexts:
                    raise ValueError(f"context {context_id!r} is already registered")
            physical = HexImplicitEulerLoss(self._rest, lame_lambda, lame_mu, density, time_step=self.time_step)
            mass = physical.lumped_mass
            if not torch.isfinite(mass).all() or (mass <= 0).any():
                raise ValueError("all physical masses, including pins, must remain positive float32")
            lam, mu = physical.lame_lambda, physical.lame_mu
            scale = torch.maximum(lam, mu)
            stiffness = mu * (3 - (mu / scale) / (lam / scale + mu / scale))
            fusion = HexFusion(self._rest, self._fixed_cpu, cell_weights=stiffness * self.cell_size**3)
            predictor = RigidPosePredictor(mass, gravity=tuple(self._gravity_cpu.tolist()))
            context = _PhysicalContext(
                specification,
                torch.stack((lam[0], mu[0], physical.density[0], damping_tensor)),
                mass,
                fusion,
                predictor,
            )
            with self._contexts_lock:
                self._contexts[context_id] = context

    def discard_context(self, context_id: str) -> None:
        """Release a context; an outstanding autograd graph retains its own factor.

        Removing registry ownership releases native predictor state immediately
        after active readers finish. PARDISO closes on its last reference, so
        an already constructed forward graph can still perform its adjoint.
        """
        with self._contexts_lock:
            self._contexts.pop(context_id)

    def close(self) -> None:
        """Release all registered native contexts and cached sparse factors."""
        with self._build_lock, self._contexts_lock:
            self._contexts.clear()

    def _lookup(self, context_ids: tuple[str, ...], batch: int) -> tuple[_PhysicalContext, ...]:
        if not isinstance(context_ids, tuple) or len(context_ids) != batch:
            raise ValueError("context_ids must be a tuple with one identifier per batch item")
        with self._contexts_lock:
            return tuple(self._contexts[name] for name in context_ids)

    def _check_positions(self, value: Tensor, name: str) -> None:
        if not isinstance(value, Tensor):
            raise ValueError(f"{name} must be a tensor")
        if value.ndim != 3 or value.shape[1:] != self.rest_positions.shape or not value.shape[0]:
            raise ValueError(f"{name} must have shape [nonempty_batch, corner_count, 3]")
        if value.dtype != torch.float32 or value.device != self.rest_positions.device:
            raise ValueError(f"{name} must use float32 on the module device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")

    def _check_previous_positions(self, positions, previous_positions, *, required):
        if previous_positions is None:
            if required:
                raise ValueError("previous_positions must supply the physical-step start positions")
            return
        self._check_positions(previous_positions, "previous_positions")
        if previous_positions.shape != positions.shape:
            raise ValueError("previous_positions must have the same batch shape as positions")

    def _check_history(self, history, positions: Tensor) -> OptimizerHistory | None:
        return check_history(
            history,
            batch=positions.shape[0],
            cell_count=len(self.cell_corner_indices),
            dtype=torch.float32,
            device=positions.device,
        )

    def prepare_inputs(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        context_ids: tuple[str, ...],
        *,
        previous_positions: Tensor,
        history: OptimizerHistory | None = None,
    ) -> LearnedHexInputs:
        """Encode mixed materials, frozen frames, the gradient feature and history.

        Args:
            positions: Candidate world corners [m], shape [B, P, 3].
            inertial_prediction: Unchanged physical Y [m], same shape.
            context_ids: One registered context identifier per object.
            previous_positions: Physical-step start [m], same shape. It anchors
                the damping term and the physical axis-change block and stays
                fixed across the inner queries of one physical step.
            history: Detached previous-query history, or None for no history
                on the whole batch (zero blocks, ``history_valid = 0``).

        Returns:
            Frozen frames, differentiable local axes, packed state, edge and
            conditioning features, plus the detached world axis gradient, the
            zero-pinned position gradient [N] and the frame tie mask.
        """
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if positions.shape != inertial_prediction.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        self._check_previous_positions(positions, previous_positions, required=True)
        contexts = self._lookup(context_ids, positions.shape[0])
        history = self._check_history(history, positions)
        return self._prepare_inputs(positions, inertial_prediction, contexts, previous_positions, history)

    def _prepare_inputs(
        self, positions: Tensor, inertial_prediction: Tensor, contexts, previous_positions: Tensor, history
    ) -> LearnedHexInputs:
        """Compose the shared schema-3 assembly with this batch's per-context energy and fusion adjoints."""

        def energy_total(candidate: Tensor, target: Tensor, previous: Tensor) -> Tensor:
            return self._energy(candidate, target, contexts, previous).total

        def project_gradient(position_gradient: Tensor) -> Tensor:
            return torch.cat(
                [context.fusion.project_gradient(position_gradient[i : i + 1]) for i, context in enumerate(contexts)]
            )

        material = torch.stack([context.material for context in contexts]).to(positions.device)
        channels = conditioning_channels(*material.unbind(-1), self.cell_size, self.time_step)
        conditioning = channels[:, None].expand(-1, len(self.cell_corner_indices), -1)
        return assemble_inputs(
            self,
            positions,
            inertial_prediction,
            previous_positions,
            energy_total=energy_total,
            project_gradient=project_gradient,
            conditioning=conditioning,
            history=history,
        )

    def energy(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        context_ids: tuple[str, ...],
        *,
        previous_positions: Tensor | None = None,
    ) -> HexLossTerms:
        """Evaluate batched full-quadrature elasticity, inertia, and damping [J].

        The stable Neo-Hookean density is finite for inverted and collapsed
        Gauss points; only nonfinite inputs raise. Positive damping requires
        ``previous_positions`` [m], the unchanged physical-step starting
        positions, with the same shape as positions.
        """
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if positions.shape != inertial_prediction.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        contexts = self._lookup(context_ids, positions.shape[0])
        self._check_previous_positions(
            positions, previous_positions, required=any(context.specification["damping"] > 0 for context in contexts)
        )
        return self._energy(positions, inertial_prediction, contexts, previous_positions)

    def _energy(self, positions: Tensor, inertial_prediction: Tensor, contexts, previous_positions) -> HexLossTerms:
        corners = positions[:, self.cell_corner_indices]
        deformation = torch.einsum("bcki,qkj->bcqij", corners - corners[:, :, :1], self.shape_gradients)
        material = torch.stack([context.material for context in contexts]).to(positions.device)
        lam, mu = material[:, 0, None, None], material[:, 1, None, None]
        density = stable_neo_hookean_density(deformation, mu, lam)
        elastic = (density * self.quadrature_weights[None, None]).sum((1, 2))
        masses = torch.stack([context.mass for context in contexts]).to(positions.device)
        step = positions.new_tensor(self.time_step)
        inertia = 0.5 * (masses[..., None] * (positions - inertial_prediction).square()).sum((1, 2)) / step.square()
        damping = torch.zeros_like(elastic)
        if any(context.specification["damping"] > 0 for context in contexts):
            difference = damping_metric_difference(
                positions, previous_positions, self.cell_corner_indices, self.shape_gradients
            )
            damping_density = material[:, 3, None, None] * difference.square().sum((-1, -2)) / (2 * step)
            damping = (damping_density * self.quadrature_weights[None, None]).sum((1, 2))
        return HexLossTerms(elastic + inertia + damping, elastic, inertia, damping)

    def energy_floor(self, context_ids: tuple[str, ...]) -> Tensor:
        """Return the detached material-aware energy floor [J], shape [B] float32.

        ``floor = c * eps32 * V * (lambda + 2 mu + eta / dt + rho h^2 / dt^2)``
        with ``V`` the total rest volume, ``eps32 = 2**-23`` and ``c`` the
        constructor's ``energy_floor_scale`` (default 1). Evidence:
        ``generated/verification/energy_floor_calibration/SUMMARY.md``
        (provisional ``c = 1``).
        """
        if not isinstance(context_ids, tuple) or not context_ids:
            raise ValueError("context_ids must be a nonempty tuple of identifiers")
        contexts = self._lookup(context_ids, len(context_ids))
        material = torch.stack([context.material for context in contexts]).to(torch.float64)
        lam, mu, rho, damping = material.unbind(-1)
        step, size = self.time_step, self.cell_size
        modulus = lam + 2 * mu + damping / step + rho * size**2 / step**2
        floor = self.energy_floor_scale * _FLOAT32_EPSILON * self.rest_volume * modulus
        return floor.to(dtype=torch.float32, device=self.rest_positions.device)

    def forward(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        context_ids: tuple[str, ...],
        *,
        fixed_positions: Tensor | None = None,
        previous_positions: Tensor,
        history: OptimizerHistory | None = None,
    ) -> LearnedHexStepOutput:
        """Make one batched proposal, fuse per material, and evaluate physical energy.

        Args:
            positions: Candidate world corners [m], shape [B, P, 3].
            inertial_prediction: Unchanged physical Y [m], same shape.
            context_ids: One registered context identifier per object.
            fixed_positions: Prescribed corners [m], shape [B, K, 3] in
                ``fixed_indices`` order; defaults to the rest positions.
            previous_positions: Unchanged physical-step start [m], same shape.
            history: Detached previous-query history or None.

        Returns:
            Fused positions, raw network outputs, frozen frames and energies,
            plus detached diagnostics: the world axis gradient feature, the
            achieved world change of the center deformation, the free-corner
            force residual norm [N] at the pre-update candidate, and the tie mask.
        """
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if positions.shape != inertial_prediction.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        self._check_previous_positions(positions, previous_positions, required=True)
        contexts = self._lookup(context_ids, positions.shape[0])
        history = self._check_history(history, positions)
        inputs = self._prepare_inputs(positions, inertial_prediction, contexts, previous_positions, history)
        prediction = self.network(inputs.local_axes, inputs.state_features, inputs.edge_features, inputs.conditioning)
        world_increment = inputs.frames @ (prediction.local_target_axes - inputs.local_axes)
        if fixed_positions is None:
            fixed_positions = self.rest_positions[self.fixed_indices][None].expand(len(contexts), -1, -1)
        if (
            not isinstance(fixed_positions, Tensor)
            or fixed_positions.shape != (len(contexts), len(self.fixed_indices), 3)
            or fixed_positions.dtype != positions.dtype
            or fixed_positions.device != positions.device
            or not torch.isfinite(fixed_positions).all()
        ):
            raise ValueError("fixed_positions must be finite [B,F,3] on the input dtype/device")
        fused = torch.cat(
            [
                context.fusion.fuse(positions[i : i + 1], world_increment[i : i + 1], fixed_positions[i : i + 1])
                for i, context in enumerate(contexts)
            ]
        )
        loss = self._energy(fused, inertial_prediction, contexts, previous_positions)
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

    def _cpu_snapshot(self, value: Tensor, name: str) -> Tensor:
        if not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a CPU float32 tensor")
        if value.shape != (len(self._rest.corner_rest_positions), 3) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite [P,3] tensor")
        return value.detach().clone()

    def prepare(self, context_id: str, positions: Tensor, velocities: Tensor, *, forces: Tensor | None = None) -> dict:
        """Snapshot a physical step with one native rigid integration and no energy.

        Inputs and tensor payloads are unbatched detached CPU float32. Positions
        use meters, velocities m/s and forces N. The payload contains only a
        context identifier and tensors, with no model, factor or native handle.
        Positive pin masses participate in momentum and inertia. Prescribed
        corners keep their input positions in the initialized candidate.
        ``physical_positions`` remains the physical-step anchor for every inner
        update. The rigid initializer never writes optimizer history.
        """
        context = self._lookup((context_id,), 1)[0]
        x = self._cpu_snapshot(positions, "positions")
        velocity = self._cpu_snapshot(velocities, "velocities")
        force = torch.zeros_like(x) if forces is None else self._cpu_snapshot(forces, "forces")
        with torch.no_grad(), context.lock:
            rigid = context.predictor.predict(x, velocity, force, self.time_step)
            fixed_positions = x[self._fixed_cpu].clone()
            base = x[None] @ rigid.rigid_delta_rotation.transpose(-1, -2) + rigid.rigid_delta_translation[:, None]
            zero_increment = x.new_zeros((1, len(self._rest.cell_corner_indices), 3, 3))
            candidate = context.fusion.fuse(base, zero_increment, fixed_positions[None])[0]
            inertial = make_inertial_prediction(
                x[None],
                velocity[None],
                self.time_step,
                explicit_acceleration=self._gravity_cpu + force / context.mass[:, None],
            )[0]
        return {
            "context_id": context_id,
            "physical_positions": x,
            "velocities": velocity,
            "candidate": candidate.detach(),
            "inertial_prediction": inertial.detach(),
            "fixed_positions": fixed_positions,
            "forces": force,
        }

    def advance(self, payload: dict) -> dict:
        """Commit candidate displacement to velocity and prepare the next step once."""
        candidate = self._cpu_snapshot(payload["candidate"], "candidate")
        previous = self._cpu_snapshot(payload["physical_positions"], "physical_positions")
        velocity = (candidate - previous) / self.time_step
        velocity[self._fixed_cpu] = 0
        return self.prepare(payload["context_id"], candidate, velocity, forces=payload.get("forces"))
