# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental heterogeneous physical contexts sharing one batched network.

Geometry, the mode-vector input schema (``target_modes`` world target vectors
per cell, see :mod:`.input_assembly`) and eight-point hex energy use batched
float32 Torch. Contexts are ordinary Python objects, excluded from module
state and DDP broadcasts; each owns a native Newton rigid predictor and its
own gravity vector.

The public API of :class:`MixedHexSolverStep` is in SI units (metres, seconds,
pascals, joules, newtons), but the objective, the fusion and the network inputs
are evaluated internally in the cell units of the normalisation law validated
in ``tests/test_scaling_invariance.py`` (``notes/ideas/idea-normalize-cells.md``):
lengths are divided by the cell size ``h``, the time step becomes one and the
stress unit ``S = mu`` makes the shear modulus one, so a context is stored as
``lambda' = lambda / mu``, ``rho' = rho h^2 / (mu dt^2)``, ``eta' = eta / (mu
dt)``, ``g' = g dt^2 / h``, ``ke' = ke / (mu h)``, ``kd' = kd / (mu h dt)`` and
``friction_epsilon' = friction_epsilon dt / h``. Energies convert back with ``S
h^3``, forces with ``S h^2`` and positions with ``h``; deformation gradients,
frames and every network input are dimensionless by construction, so two
scenes related by the law produce identical network inputs and outputs that
differ exactly by the length factor. Because the physical fusion weights of a
homogeneous body are uniform and a weighted least-squares fit is invariant
under a uniform rescaling of its weights, one unit-weight PARDISO factor on
the unit grid serves every context; registering a context no longer builds a
factor.

Frames are the closest proper rotations of the cell-center deformation with the
clamped-face tie-break from :mod:`frames`. Inverted and collapsed candidates
are accepted: the Newton stable Neo-Hookean law is finite for every finite
shape, no geometry backtracking or acceptance scaling exists, and only
nonfinite inputs raise. The gradient input is the detached position gradient of
the complete normalised objective projected through the shared fusion adjoint,
normalized with the LeCO convention in :mod:`features`; its log RMS is
therefore dimensionless. The single-material :class:`.solver_step.LearnedHexSolverStep`
expresses its gradient feature in the same unit ``S h^3``, so both steps feed
one network identical inputs and share the ``LearnedHexInputs`` unit
convention: ``axis_gradient_world`` in ``S h^3``, ``position_gradient`` in
newtons.

Contact (``notes/contact-design-20260927.md``) enters through per-context
static partners (:class:`.contact_scene.ContactPartners`): :meth:`prepare`
detects candidate pairs once per physical step on the step-start shape in SI,
the padded pair batch is passed back to :meth:`energy` and :meth:`forward`,
the penalty energy of :mod:`.contact_energy` joins the objective, and the
network receives schema-4 contact tokens and conditioning channels. A context
without partners is contact-free and reproduces the contact-less objective
exactly.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import NamedTuple

import numpy as np
import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor, nn  # noqa: TID253

from .contact_energy import contact_energy, contact_penetration
from .contact_features import build_contact_tokens
from .contact_geometry import exposed_face_samples, sample_normals, sample_points
from .contact_scene import ContactPartners, detect_contacts
from .damping import damping_metric_difference
from .data import VoxelGridData, generate_cuboid
from .features import (
    CONDITIONING_DIM,
    CONTACT_TOKEN_DIM,
    EDGE_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    conditioning_channels,
    contact_ratios,
    state_feature_dim,
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
    target_vectors,
)
from .network import IntrinsicSolverNetwork
from .rigid_predictor import RigidPosePredictor

__all__ = ["MixedHexSolverStep", "OptimizerHistory"]

_FLOAT32_EPSILON = 2.0**-23
"""Unit roundoff of float32 used by the material-aware energy floor."""

_CONTACT_BATCH_KEYS = ("sample_index", "kind", "partner_point", "partner_normal", "partner_radius", "mask")
"""Entries of the padded contact pair batch accepted by :meth:`MixedHexSolverStep.energy`."""


@dataclass
class _PhysicalContext:
    """One registered material: the SI record behind the API and its normalised copy behind the objective."""

    specification: dict[str, float | tuple[float, float, float]]
    """SI material, damping and gravity as registered; ``context_specs`` returns copies."""
    material: Tensor
    """SI ``(lambda, mu, rho, eta)`` float32 [4]; the energy floor and the stress unit ``S = mu`` read it."""
    unit_material: Tensor
    """Normalised ``(lambda / mu, 1, rho h^2 / (mu dt^2), eta / (mu dt))`` float32 [4]."""
    mass: Tensor
    """SI lumped corner masses [kg], shape [P], for the rigid predictor and the force acceleration in prepare."""
    unit_mass: Tensor
    """Normalised lumped corner masses ``rho' / 8`` per incident cell, shape [P]."""
    predictor: RigidPosePredictor
    contact: ContactPartners
    """SI static partners used by contact detection in prepare."""
    unit_contact: Tensor
    """Normalised ``(ke / (mu h), kd / (mu h dt), mu_f)`` float32 [3] used by the contact energy."""
    conditioning: Tensor
    """Dimensionless conditioning channels of :func:`.features.conditioning_channels`, shape [CONDITIONING_DIM]."""
    contact_ratios: Tensor
    """Dimensionless ``(kappa, beta, mu_f)`` of :func:`.features.contact_ratios` for the contact tokens, shape [3]."""
    gravity: tuple[float, float, float]
    """SI gravity [m/s^2] of this context as float64-derived floats; the conditioning and the predictor read it."""
    gravity_cpu: Tensor
    """Float32 CPU copy of ``gravity`` that the inertial prediction of :meth:`MixedHexSolverStep.prepare` adds."""
    lock: threading.RLock = field(default_factory=threading.RLock)


class _UnitGeometry(NamedTuple):
    """Cell-unit view of the canonical grid that :func:`.input_assembly.assemble_inputs` consumes."""

    cell_corner_indices: Tensor
    center_gradients: Tensor
    rest_centers: Tensor
    cell_size: float
    reference_corners: Tensor
    fixed_indices: Tensor
    boundary_features: Tensor
    network: IntrinsicSolverNetwork
    target_modes: int


def _gravity_vector(gravity) -> Tensor:
    """Return a finite SI gravity three-vector as a detached float64 CPU tensor, or raise ValueError."""
    try:
        vector = torch.as_tensor(gravity, dtype=torch.float64, device="cpu").detach().clone()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("gravity must be a finite three-vector") from error
    if vector.shape != (3,) or not torch.isfinite(vector).all():
        raise ValueError("gravity must be a finite three-vector")
    return vector


def _positive_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _nonnegative_integer(name: str, value, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class MixedHexSolverStep(nn.Module):
    """Apply one learned update to objects with distinct homogeneous materials.

    Experimental. All objects share a canonical grid, pin indices and
    timestep; every context carries its own gravity (the constructor's value
    is the default for contexts registered without one). Each batch evaluates
    the shared network exactly once. Register material contexts on CPU before
    using their IDs; registration and physical preparation never read
    trainable weights. Contexts may be created by worker threads while another
    batch runs. Wrap this module with DDP using ``broadcast_buffers=False``
    and checkpoint ``context_specs`` separately.

    Inputs and outputs are SI; the objective, the fusion and the network inputs
    are evaluated in cell units (module docstring). Positions, the inertial
    prediction, the physical-step start and the contact partner geometry are
    divided by ``h`` on the way in; energies are multiplied by ``S h^3`` (``S =
    mu`` of the context), the force residual by ``S h^2`` and fused positions
    by ``h`` on the way out, with the prescribed corners re-assigned exactly.
    The world target gradient (``axis_gradient_world`` of the inputs and the
    output, and hence the optimizer history) and the achieved target update
    stay in normalised units: the history is RMS-normalised against the
    current gradient when it is consumed, so only consistency within an object
    matters, and the achieved update is a difference of target vectors
    (:func:`.input_assembly.target_vectors` on the unit grid), which is
    dimensionless in either space. ``position_gradient`` of the inputs is
    converted to newtons like the force residual.

    Every cell carries ``target_modes`` world target vectors: the three centre
    axes of ``F`` (``target_modes = 3``, the legacy nine-value schema) or those
    plus the four warping vectors of :mod:`.hex_modes` (``target_modes = 7``).
    The network must match: ``features.state_feature_dim(target_modes)``
    state inputs (five ``[3, target_modes]`` matrix blocks in the receiving
    frame, boundary flags, log gradient RMS and the history flag), the
    :data:`features.CONDITIONING_DIM` dimensionless conditioning channels
    (including the three contact channels), 24 edge inputs and the same
    ``target_modes``. The shared unit-grid fusion, the history blocks and the
    achieved update use the same mode count. Frames are the closest proper
    rotations of the cell-center deformation (the first three target vectors);
    ties are broken with the reference built from three prescribed corners of
    the clamped face when :func:`frames.select_reference_corners` finds them,
    otherwise the plain formula is kept. The frame decomposition and the
    gradient feature are frozen for the query; the network, local axes, fusion
    and energy remain differentiable to every network parameter.

    The rigid predictor only initializes the candidate. The physical inertial
    target stays unchanged through learned updates. Inverted or collapsed
    candidates are evaluated with the stable Neo-Hookean law and are never
    rejected or shortened; nonfinite inputs raise. Energy-descent acceptance is
    not implemented.

    Contact uses one surface sample per exposed face (radius ``contact_radius``,
    default ``0.5 h``) against the static partners registered with each
    context; a partner pairs with a sample only when its normal opposes the
    sample's face normal. :meth:`prepare` writes the frozen pair list of a
    physical step into the payload; callers collate it into the padded batch that :meth:`energy`
    and :meth:`forward` accept through ``contact``. When the network was built
    with ``contact_tokens=True`` the step also builds its per-cell contact
    tokens (``contact_tokens_per_cell`` slots).

    Args:
        rest: Canonical cubic hexahedral rest grid [m].
        fixed_indices: Unique prescribed corner indices, at least one.
        network: Shared float32 network with the revised schema and
            ``target_modes`` modes, on CPU or CUDA.
        time_step: Positive physical timestep [s].
        gravity: Default world acceleration [m/s^2] for contexts registered
            without their own ``gravity``.
        energy_floor_scale: Positive multiplier ``c`` of the material-aware
            energy floor returned by :meth:`energy_floor`.
        contact_radius: Surface sample radius r [m]; None means ``0.5 * cell_size``.
        contact_max_pairs: Largest number of static-point pairs kept per sample.
        contact_tokens_per_cell: Token slots M per cell for the contact encoder.
        contact_friction_epsilon: IPC friction smoothing band as a fraction of the time step.
        target_modes: Target vectors per cell, 3 (affine axes, the default)
            or 7 (axes plus warping vectors); the network must agree.
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
        contact_radius: float | None = None,
        contact_max_pairs: int = 4,
        contact_tokens_per_cell: int = 24,
        contact_friction_epsilon: float = 1e-2,
        target_modes: int = 3,
    ):
        super().__init__()
        if isinstance(target_modes, bool) or target_modes not in (3, 7):
            raise ValueError("target_modes must be 3 (affine axes) or 7 (axes plus warping vectors)")
        self.target_modes = int(target_modes)
        _positive_scalar("time_step", time_step)
        _positive_scalar("energy_floor_scale", energy_floor_scale)
        self.contact_radius = (
            0.5 * rest.cell_size if contact_radius is None else _positive_scalar("contact_radius", contact_radius)
        )
        self.contact_max_pairs = _nonnegative_integer("contact_max_pairs", contact_max_pairs)
        self.contact_tokens_per_cell = _nonnegative_integer(
            "contact_tokens_per_cell", contact_tokens_per_cell, minimum=1
        )
        self.contact_friction_epsilon = _positive_scalar("contact_friction_epsilon", contact_friction_epsilon)
        if network.cell_counts != rest.cell_counts:
            raise ValueError("network cell_counts must match the rest grid")
        schema = (network.state_feature_dim, network.conditioning_dim, network.edge_input_dim, network.target_modes)
        expected_state = state_feature_dim(self.target_modes)
        if schema != (expected_state, CONDITIONING_DIM, EDGE_FEATURE_DIM, self.target_modes):
            raise ValueError(
                f"network must use the revised schema {FEATURE_SCHEMA_VERSION} with target_modes "
                f"{self.target_modes}: {expected_state} state, {CONDITIONING_DIM} conditioning and "
                f"{EDGE_FEATURE_DIM} edge inputs and {self.target_modes} target modes, got "
                f"{schema[0]}/{schema[1]}/{schema[2]} with {schema[3]} modes; legacy 38/5, 86/6 and 61/9 networks "
                "are not supported"
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
        gravity_tensor = _gravity_vector(gravity)
        self._rest = rest
        self._fixed_cpu = torch.tensor(fixed.copy(), dtype=torch.long)
        # The default gravity of register_context: the float64 tuple feeds the conditioning (the same value
        # LearnedHexSolverStep forms, so the gravity channel agrees between the two steps bit for bit).
        self.gravity = tuple(float(value) for value in gravity_tensor.tolist())
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
        # CPU copy for detection in prepare(); the buffers follow the module device for the energy and tokens.
        self.face_samples = exposed_face_samples(rest)
        self.register_buffer("face_corners", self.face_samples.corners.clone())
        self.register_buffer("face_cell_index", self.face_samples.cell_index.clone())
        # Cell-unit copy of the grid (same topology, lengths divided by h) behind the objective, the fusion
        # and the network inputs. Its buffers derive from the rest grid alone and stay out of the state_dict.
        # The corners are generated on the unit lattice from the scaled origin rather than divided by h, so the
        # canonical-grid check of HexImplicitEulerLoss (atol 1e-12) passes for any origin the SI grid passes.
        h, dt = self.cell_size, self.time_step
        unit_lattice = generate_cuboid(rest.cell_counts, cell_size=1.0, origin=tuple(rest.corner_rest_positions[0] / h))
        self._unit_rest = replace(
            rest,
            cell_size=1.0,
            corner_rest_positions=unit_lattice.corner_rest_positions,
            cell_rest_centers=rest.cell_rest_centers / h,
            cell_velocity=rest.cell_velocity * (dt / h),
        )
        unit_geometry = HexImplicitEulerLoss(self._unit_rest, 0.0, 1.0, 1.0, 1.0)
        self.register_buffer("unit_shape_gradients", unit_geometry.shape_gradients, persistent=False)
        self.register_buffer("unit_quadrature_weights", unit_geometry.quadrature_weights, persistent=False)
        self.register_buffer(
            "unit_rest_centers", torch.tensor(self._unit_rest.cell_rest_centers, dtype=torch.float32), persistent=False
        )
        self.register_buffer("unit_center_gradients", signs / 4, persistent=False)
        # Gravity has no term in the objective (it sits in the SI inertial prediction of prepare), so only the
        # conditioning sees it, per context, through the group |g| dt^2 / h that features.conditioning_channels forms.
        self._unit_radius = self.contact_radius / h
        self._unit_friction_epsilon = self.contact_friction_epsilon * dt / h
        # One factor serves every context: the physical weights ``stiffness h^3`` of a homogeneous body are
        # uniform, and a weighted least-squares fit is invariant under a uniform rescaling of its weights, so
        # unit weights on the unit grid reproduce every former per-material fit (tests/test_scaling_invariance.py).
        self._fusion = HexFusion(self._unit_rest, self._fixed_cpu, target_modes=self.target_modes)
        self.to(device=device)

    @property
    def context_specs(self) -> dict[str, dict[str, float | tuple[float, float, float]]]:
        """Return independent material [Pa, kg/m^3], damping [Pa*s] and gravity [m/s^2] specifications.

        Each entry has the keyword arguments of :meth:`register_context` except
        ``contact``: ``lame_lambda``, ``lame_mu``, ``density``, ``damping`` and
        the ``gravity`` three-tuple, so ``register_context(name, **spec,
        contact=...)`` rebuilds the context.
        """
        with self._contexts_lock:
            return {name: context.specification.copy() for name, context in self._contexts.items()}

    def register_context(
        self,
        context_id: str,
        *,
        lame_lambda: float,
        lame_mu: float,
        density: float,
        damping: float = 0.0,
        contact: ContactPartners | None = None,
        gravity=None,
    ) -> None:
        """Build one CPU material/predictor context without consulting weights.

        The SI material is kept for the API (``context_specs``, the energy
        floor, the rigid predictor) and stored a second time in cell units for
        the objective; the dimensionless conditioning and contact ratios are
        computed once here. Gravity is a per-context quantity: it drives the
        context's rigid predictor and the inertial prediction of
        :meth:`prepare` and enters the conditioning channel ``log1p(|g| dt^2 /
        h)``. No sparse factor is built: every context shares the module's
        unit fusion factor.

        Args:
            context_id: Unique nonempty identifier for replay payloads.
            lame_lambda: Nonnegative first Lamé parameter [Pa].
            lame_mu: Positive shear modulus [Pa].
            density: Positive rest density [kg/m^3].
            damping: Nonnegative absolute damping coefficient [Pa*s], held
                fixed for this context.
            contact: Static contact partners of this context's scene; None
                registers :meth:`.ContactPartners.contact_free` (no plane, no
                points, zero coefficients).
            gravity: World acceleration [m/s^2] of this context as a finite
                three-vector (sequence or tensor); None uses the constructor's
                ``gravity``.

        Raises:
            ValueError: If the identifier is taken or empty, a material value
                is outside its physical range, ``contact`` has the wrong type
                or ``gravity`` is not a finite three-vector.
        """
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("context_id must be a nonempty string")
        if contact is None:
            contact = ContactPartners.contact_free()
        elif not isinstance(contact, ContactPartners):
            raise ValueError("contact must be ContactPartners or None")
        specification = {"lame_lambda": lame_lambda, "lame_mu": lame_mu, "density": density, "damping": damping}
        for name, value in specification.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            if value < 0 or (name not in ("lame_lambda", "damping") and value == 0):
                raise ValueError(f"{name} is outside the physical range")
        specification = {name: float(value) for name, value in specification.items()}
        gravity_tensor = _gravity_vector(self.gravity if gravity is None else gravity)
        gravity_vector = tuple(float(value) for value in gravity_tensor.tolist())
        specification["gravity"] = gravity_vector
        damping_tensor = torch.tensor(damping, dtype=torch.float32)
        if not torch.isfinite(damping_tensor):
            raise ValueError("damping must remain finite in float32")
        h, dt = self.cell_size, self.time_step
        # Serialize native construction independently of forward lookups. Warp
        # setup is never performed under the registry lock.
        with self._build_lock:
            with self._contexts_lock:
                if context_id in self._contexts:
                    raise ValueError(f"context {context_id!r} is already registered")
            physical = HexImplicitEulerLoss(self._rest, lame_lambda, lame_mu, density, time_step=dt)
            mass = physical.lumped_mass
            if not torch.isfinite(mass).all() or (mass <= 0).any():
                raise ValueError("all physical masses, including pins, must remain positive float32")
            normalised = HexImplicitEulerLoss(
                self._unit_rest,
                lame_lambda / lame_mu,
                1.0,
                density * h**2 / (lame_mu * dt**2),
                time_step=1.0,
                damping=damping / (lame_mu * dt),
            )
            unit_mass = normalised.lumped_mass
            if not torch.isfinite(unit_mass).all() or (unit_mass <= 0).any():
                raise ValueError("all normalised masses must remain positive float32")
            material = torch.stack((physical.lame_lambda[0], physical.lame_mu[0], physical.density[0], damping_tensor))
            unit_material = torch.stack(
                (normalised.lame_lambda[0], normalised.lame_mu[0], normalised.density[0], normalised.damping[0])
            )
            coefficients = torch.tensor([[contact.ke, contact.kd, contact.mu]], dtype=torch.float32)
            unit_contact = torch.tensor(
                [contact.ke / (lame_mu * h), contact.kd / (lame_mu * h * dt), contact.mu], dtype=torch.float32
            )
            if not torch.isfinite(unit_contact).all():
                raise ValueError("normalised contact coefficients must remain finite in float32")
            conditioning = conditioning_channels(
                *material[None].unbind(-1),
                h,
                dt,
                gravity_vector,
                contact_ke=coefficients[:, 0],
                contact_kd=coefficients[:, 1],
                contact_mu=coefficients[:, 2],
            )[0]
            ratios = contact_ratios(material[None, 0], material[None, 1], *coefficients.unbind(-1), h, dt)[0]
            predictor = RigidPosePredictor(mass, gravity=gravity_vector)
            context = _PhysicalContext(
                specification,
                material,
                unit_material,
                mass,
                unit_mass,
                predictor,
                contact,
                unit_contact,
                conditioning,
                ratios,
                gravity_vector,
                gravity_tensor.to(torch.float32),
            )
            with self._contexts_lock:
                self._contexts[context_id] = context

    def discard_context(self, context_id: str) -> None:
        """Release a context's native predictor state after active readers finish.

        The shared fusion factor belongs to the module and is unaffected; an
        outstanding autograd graph keeps its own reference to it.
        """
        with self._contexts_lock:
            self._contexts.pop(context_id)

    def close(self) -> None:
        """Release all registered native contexts; the shared fusion factor stays with the module."""
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
            modes=self.target_modes,
        )

    def _check_contact(self, contact, positions: Tensor) -> dict[str, Tensor] | None:
        """Validate the padded contact pair batch; return None when it carries no pair (Q = 0)."""
        if contact is None:
            return None
        if not isinstance(contact, dict):
            raise ValueError("contact must be a dict of padded pair tensors or None")
        missing = [name for name in _CONTACT_BATCH_KEYS if name not in contact]
        if missing:
            raise ValueError(f"contact is missing entries {missing}")
        batch = positions.shape[0]
        sample_index = contact["sample_index"]
        if not isinstance(sample_index, Tensor) or sample_index.ndim != 2 or sample_index.shape[0] != batch:
            raise ValueError("contact['sample_index'] must be an integer tensor of shape [B, Q]")
        pairs = int(sample_index.shape[1])
        if pairs == 0:
            return None
        shapes = {
            "sample_index": (batch, pairs),
            "kind": (batch, pairs),
            "partner_point": (batch, pairs, 3),
            "partner_normal": (batch, pairs, 3),
            "partner_radius": (batch, pairs),
            "mask": (batch, pairs),
        }
        checked = {}
        for name, shape in shapes.items():
            value = contact[name]
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"contact[{name!r}] must be a tensor of shape {list(shape)}")
            if value.device != positions.device:
                raise ValueError(f"contact[{name!r}] must be on the module device")
            if name in ("sample_index", "kind"):
                if value.dtype not in (torch.int64, torch.int32):
                    raise ValueError(f"contact[{name!r}] must be an int64 or int32 tensor")
                value = value.to(torch.int64)
            elif name == "mask":
                if value.dtype != torch.bool:
                    raise ValueError("contact['mask'] must be a boolean tensor")
            elif value.dtype != torch.float32:
                raise ValueError(f"contact[{name!r}] must be a float32 tensor")
            checked[name] = value.detach()
        mask = checked["mask"]
        for name in ("partner_point", "partner_normal", "partner_radius"):
            value = checked[name]
            valid = mask[..., None] if value.ndim == 3 else mask
            if not torch.isfinite(torch.where(valid, value, torch.zeros_like(value))).all():
                raise ValueError(f"contact[{name!r}] must be finite for valid pairs")
        return checked

    # -- unit conversion -------------------------------------------------------------------------------------

    def _to_unit(self, *tensors: Tensor | None) -> tuple[Tensor | None, ...]:
        """Divide SI lengths [m] by the cell size; None passes through."""
        return tuple(None if value is None else value / self.cell_size for value in tensors)

    def _to_unit_contact(self, contact: dict[str, Tensor] | None) -> dict[str, Tensor] | None:
        """Return the validated contact batch with its partner geometry in cell units."""
        if contact is None:
            return None
        h = self.cell_size
        return {
            **contact,
            "partner_point": contact["partner_point"] / h,
            "partner_radius": contact["partner_radius"] / h,
        }

    def _from_unit_positions(self, unit_positions: Tensor, fixed_positions: Tensor) -> Tensor:
        """Return SI corners ``h X'`` with the prescribed rows set to ``fixed_positions`` exactly.

        Scaling back can move a prescribed row by one float32 ulp; the
        prescribed positions always win exactly, as they do inside the fusion.
        """
        fixed = self._fixed_cpu if unit_positions.device.type == "cpu" else self.fixed_indices
        return (unit_positions * self.cell_size).index_copy(1, fixed, fixed_positions)

    def _stress_unit(self, contexts, device) -> Tensor:
        """Return ``S = mu`` [Pa] of every context, float32 [B]."""
        return torch.stack([context.material[1] for context in contexts]).to(device)

    def _from_unit_energy(self, terms: HexLossTerms, contexts, device) -> HexLossTerms:
        """Scale the normalised energy parts by ``S h^3`` to joules; the total is their exact float32 sum."""
        scale = self._stress_unit(contexts, device) * self.cell_size**3
        elastic, inertia, damping, contact = (term * scale for term in terms[1:])
        return HexLossTerms(elastic + inertia + damping + contact, elastic, inertia, damping, contact)

    def _unit_geometry(self) -> _UnitGeometry:
        """Return the cell-unit grid view for :func:`.input_assembly.assemble_inputs`."""
        return _UnitGeometry(
            self.cell_corner_indices,
            self.unit_center_gradients,
            self.unit_rest_centers,
            1.0,
            self.reference_corners,
            self.fixed_indices,
            self.boundary_features,
            self.network,
            self.target_modes,
        )

    def _sample_positions(self, positions: Tensor) -> Tensor:
        """Return the exposed-face sample centroids, shape [B, S, 3], differentiable in positions."""
        return sample_points(positions, self.face_corners)

    # -- network inputs --------------------------------------------------------------------------------------

    def prepare_inputs(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        context_ids: tuple[str, ...],
        *,
        previous_positions: Tensor,
        history: OptimizerHistory | None = None,
        contact: dict | None = None,
    ) -> LearnedHexInputs:
        """Encode mixed materials, frozen frames, the gradient feature, history and contact.

        The SI inputs are converted to cell units and the assembly runs there,
        so the returned frames, axes, state, edge and conditioning features are
        exactly what the network consumes. ``axis_gradient_world`` is the
        fusion-projected gradient of the normalised objective in units of ``S
        h^3`` (the quantity that the optimizer history carries);
        ``position_gradient`` is the zero-pinned gradient of the SI objective
        in newtons (the normalised gradient times ``S h^2``), the same
        conventions as :class:`.solver_step.LearnedHexSolverStep`.

        Args:
            positions: Candidate world corners [m], shape [B, P, 3].
            inertial_prediction: Unchanged physical Y [m], same shape.
            context_ids: One registered context identifier per object.
            previous_positions: Physical-step start [m], same shape. It anchors
                the damping term, the contact friction and the physical
                axis-change block and stays fixed across the inner queries of
                one physical step.
            history: Detached previous-query history in the normalised units
                returned by :meth:`forward`, or None for no history on the
                whole batch (zero blocks, ``history_valid = 0``).
            contact: Padded contact pair batch in SI (``sample_index`` [B, Q],
                ``kind`` [B, Q], ``partner_point`` [B, Q, 3], ``partner_normal``
                [B, Q, 3], ``partner_radius`` [B, Q], ``mask`` [B, Q]) collated
                from :meth:`prepare` payloads, or None; ``Q = 0`` means no contact.

        Returns:
            Frozen frames, differentiable local axes [B, C, 3, target_modes],
            packed state, edge and conditioning features, plus the detached
            normalised world target gradient (same shape), the zero-pinned
            position gradient [N], the frame tie mask and, when the network
            consumes contact tokens, the detached tokens and their mask.
        """
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if positions.shape != inertial_prediction.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        self._check_previous_positions(positions, previous_positions, required=True)
        contexts = self._lookup(context_ids, positions.shape[0])
        history = self._check_history(history, positions)
        contact = self._to_unit_contact(self._check_contact(contact, positions))
        unit_positions, unit_prediction, unit_previous = self._to_unit(
            positions, inertial_prediction, previous_positions
        )
        return self._prepare_inputs(unit_positions, unit_prediction, contexts, unit_previous, history, contact)

    def _prepare_inputs(
        self,
        unit_positions: Tensor,
        unit_prediction: Tensor,
        contexts,
        unit_previous: Tensor,
        history,
        unit_contact: dict[str, Tensor] | None = None,
    ) -> LearnedHexInputs:
        """Compose the shared assembly in cell units with this batch's normalised energy, the shared adjoint and contact."""

        def energy_total(candidate: Tensor, target: Tensor, previous: Tensor) -> Tensor:
            return self._unit_energy(candidate, target, contexts, previous, unit_contact).total

        device = unit_positions.device
        cells = len(self.cell_corner_indices)
        conditioning = torch.stack([context.conditioning for context in contexts]).to(device)
        inputs = assemble_inputs(
            self._unit_geometry(),
            unit_positions,
            unit_prediction,
            unit_previous,
            energy_total=energy_total,
            project_gradient=self._fusion.project_gradient,
            conditioning=conditioning[:, None].expand(-1, cells, -1),
            history=history,
        )
        force_unit = self._stress_unit(contexts, device) * self.cell_size**2
        inputs = inputs._replace(position_gradient=inputs.position_gradient * force_unit[:, None, None])
        if not self.network.contact_tokens:
            return inputs
        ratios = torch.stack([context.contact_ratios for context in contexts]).to(device)
        tokens, mask = self._contact_tokens(inputs.frames, unit_positions, unit_previous, unit_contact, ratios)
        return inputs._replace(contact_tokens=tokens, contact_mask=mask)

    def _contact_tokens(
        self, frames: Tensor, unit_positions: Tensor, unit_previous: Tensor, unit_contact, ratios: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Build the detached per-cell contact tokens (cell units) for a network with the contact flag."""
        batch, cells = frames.shape[:2]
        if unit_contact is None:
            # One all-masked slot keeps the encoder in the graph (its output is exactly zero), so
            # every parameter still receives a gradient on contact-free batches, as DDP requires.
            tokens = frames.new_zeros((batch, cells, 1, CONTACT_TOKEN_DIM))
            return tokens, torch.zeros((batch, cells, 1), dtype=torch.bool, device=frames.device)
        return build_contact_tokens(
            frames=frames,
            cell_centers=unit_positions[:, self.cell_corner_indices].mean(-2),
            cell_size=1.0,
            radius=self._unit_radius,
            face_cell_index=self.face_cell_index,
            sample_positions=self._sample_positions(unit_positions),
            sample_start_positions=self._sample_positions(unit_previous),
            sample_index=unit_contact["sample_index"],
            kind=unit_contact["kind"],
            partner_point=unit_contact["partner_point"],
            partner_normal=unit_contact["partner_normal"],
            partner_radius=unit_contact["partner_radius"],
            pair_mask=unit_contact["mask"],
            contact_kappa=ratios[:, 0],
            contact_beta=ratios[:, 1],
            contact_mu=ratios[:, 2],
            tokens_per_cell=self.contact_tokens_per_cell,
        )

    # -- objective -------------------------------------------------------------------------------------------

    def energy(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        context_ids: tuple[str, ...],
        *,
        previous_positions: Tensor | None = None,
        contact: dict | None = None,
    ) -> HexLossTerms:
        """Evaluate batched full-quadrature elasticity, inertia, damping and contact [J].

        The SI inputs are converted to cell units, the normalised objective is
        evaluated and every term is scaled back by ``S h^3``, so the result
        equals the SI objective to float32 rounding. The stable Neo-Hookean
        density is finite for inverted and collapsed Gauss points; only
        nonfinite inputs raise. Positive damping or a contact batch with pairs
        requires ``previous_positions`` [m], the unchanged physical-step
        starting positions, with the same shape as positions. ``contact`` is
        the padded pair batch described in :meth:`prepare_inputs`; None or
        ``Q = 0`` means no contact term and a zero ``contact`` entry in the
        returned terms.
        """
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if positions.shape != inertial_prediction.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        contexts = self._lookup(context_ids, positions.shape[0])
        contact = self._check_contact(contact, positions)
        self._check_previous_positions(
            positions,
            previous_positions,
            required=contact is not None or any(context.specification["damping"] > 0 for context in contexts),
        )
        return self._energy(positions, inertial_prediction, contexts, previous_positions, contact)

    def _energy(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        contexts,
        previous_positions,
        contact: dict[str, Tensor] | None = None,
    ) -> HexLossTerms:
        """Return the SI energy terms [J] of validated SI inputs through the normalised objective."""
        unit_positions, unit_prediction, unit_previous = self._to_unit(
            positions, inertial_prediction, previous_positions
        )
        terms = self._unit_energy(
            unit_positions, unit_prediction, contexts, unit_previous, self._to_unit_contact(contact)
        )
        return self._from_unit_energy(terms, contexts, positions.device)

    def _unit_energy(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        contexts,
        previous_positions,
        contact: dict[str, Tensor] | None = None,
    ) -> HexLossTerms:
        """Evaluate the objective in cell units (``h' = dt' = mu' = 1``); inputs and outputs are normalised."""
        corners = positions[:, self.cell_corner_indices]
        deformation = torch.einsum("bcki,qkj->bcqij", corners - corners[:, :, :1], self.unit_shape_gradients)
        material = torch.stack([context.unit_material for context in contexts]).to(positions.device)
        lam, mu = material[:, 0, None, None], material[:, 1, None, None]
        density = stable_neo_hookean_density(deformation, mu, lam)
        elastic = (density * self.unit_quadrature_weights[None, None]).sum((1, 2))
        masses = torch.stack([context.unit_mass for context in contexts]).to(positions.device)
        # The unit time step makes the inertia and damping denominators one.
        inertia = 0.5 * (masses[..., None] * (positions - inertial_prediction).square()).sum((1, 2))
        damping = torch.zeros_like(elastic)
        if any(context.specification["damping"] > 0 for context in contexts):
            difference = damping_metric_difference(
                positions, previous_positions, self.cell_corner_indices, self.unit_shape_gradients
            )
            damping_density = material[:, 3, None, None] * difference.square().sum((-1, -2)) / 2
            damping = (damping_density * self.unit_quadrature_weights[None, None]).sum((1, 2))
        contact_term = torch.zeros_like(elastic)
        if contact is not None:
            coefficients = torch.stack([context.unit_contact for context in contexts]).to(positions.device)
            contact_term = contact_energy(
                self._sample_positions(positions),
                self._sample_positions(previous_positions),
                contact["sample_index"],
                contact["partner_point"],
                contact["partner_normal"],
                contact["mask"],
                radius=self._unit_radius,
                ke=coefficients[:, 0],
                kd=coefficients[:, 1],
                mu=coefficients[:, 2],
                time_step=1.0,
                friction_epsilon=self._unit_friction_epsilon,
            )
        return HexLossTerms(elastic + inertia + damping + contact_term, elastic, inertia, damping, contact_term)

    def _contact_max_penetration(self, unit_positions: Tensor, unit_contact: dict[str, Tensor] | None) -> Tensor:
        """Return the deepest penetration over the frozen pairs in units of r, detached, shape [B]."""
        if unit_contact is None:
            return unit_positions.new_zeros(unit_positions.shape[0])
        depth = contact_penetration(
            self._sample_positions(unit_positions.detach()),
            unit_contact["sample_index"],
            unit_contact["partner_point"],
            unit_contact["partner_normal"],
            unit_contact["mask"],
            radius=self._unit_radius,
        )
        return depth.amax(dim=1) / self._unit_radius

    def energy_floor(self, context_ids: tuple[str, ...]) -> Tensor:
        """Return the detached material-aware energy floor [J], shape [B] float32.

        ``floor = c * eps32 * V * (lambda + 2 mu + eta / dt + rho h^2 / dt^2)``
        with ``V`` the total rest volume, ``eps32 = 2**-23`` and ``c`` the
        constructor's ``energy_floor_scale`` (default 1). The SI formula is
        kept: it equals ``S h^3`` times a sum of the dimensionless groups, so
        the ratio of an energy to its floor is the same in both spaces.
        Evidence: ``generated/verification/energy_floor_calibration/SUMMARY.md``
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

    # -- learned update --------------------------------------------------------------------------------------

    def forward(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        context_ids: tuple[str, ...],
        *,
        fixed_positions: Tensor | None = None,
        previous_positions: Tensor,
        history: OptimizerHistory | None = None,
        contact: dict | None = None,
    ) -> LearnedHexStepOutput:
        """Make one batched proposal, fuse with the shared factor, and evaluate the physical energy.

        Args:
            positions: Candidate world corners [m], shape [B, P, 3].
            inertial_prediction: Unchanged physical Y [m], same shape.
            context_ids: One registered context identifier per object.
            fixed_positions: Prescribed corners [m], shape [B, K, 3] in
                ``fixed_indices`` order; defaults to the rest positions.
            previous_positions: Unchanged physical-step start [m], same shape.
            history: Detached previous-query history (normalised units, as
                returned by this method) or None.
            contact: Padded contact pair batch in SI as in :meth:`prepare_inputs`, or None.

        Returns:
            Fused positions [m] (prescribed rows exactly ``fixed_positions``),
            raw network outputs [B, C, 3, target_modes], frozen frames and
            energies [J], plus detached diagnostics: the normalised world
            target gradient feature [B, C, 3, target_modes], the achieved
            world change of the target vectors on the unit grid (same shape,
            dimensionless), the free-corner force residual norm [N] at the
            pre-update candidate, the tie mask, the contact energy [J] and the
            deepest penetration in units of r at the fused positions (both
            zero without contact pairs).
        """
        self._check_positions(positions, "positions")
        self._check_positions(inertial_prediction, "inertial_prediction")
        if positions.shape != inertial_prediction.shape:
            raise ValueError("positions and inertial_prediction must share a batch shape")
        self._check_previous_positions(positions, previous_positions, required=True)
        contexts = self._lookup(context_ids, positions.shape[0])
        history = self._check_history(history, positions)
        contact = self._check_contact(contact, positions)
        unit_contact = self._to_unit_contact(contact)
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
        unit_positions, unit_prediction, unit_previous, unit_fixed = self._to_unit(
            positions, inertial_prediction, previous_positions, fixed_positions
        )
        inputs = self._prepare_inputs(unit_positions, unit_prediction, contexts, unit_previous, history, unit_contact)
        if self.network.contact_tokens:
            prediction = self.network(
                inputs.local_axes,
                inputs.state_features,
                inputs.edge_features,
                inputs.conditioning,
                contact_tokens=inputs.contact_tokens,
                contact_mask=inputs.contact_mask,
            )
        else:
            prediction = self.network(
                inputs.local_axes, inputs.state_features, inputs.edge_features, inputs.conditioning
            )
        world_increment = inputs.frames @ (prediction.local_target_axes - inputs.local_axes)
        fused = self._from_unit_positions(
            self._fusion.fuse(unit_positions, world_increment, unit_fixed), fixed_positions
        )
        # The objective and the diagnostics are evaluated at the returned SI positions through the same path as
        # energy(), so a caller recomputing energy(output.positions, ...) reproduces output.loss exactly.
        loss = self._energy(fused, inertial_prediction, contexts, previous_positions, contact)
        with torch.no_grad():
            (unit_fused,) = self._to_unit(fused.detach())
            geometry = self._unit_geometry()
            achieved = target_vectors(geometry, unit_fused) - target_vectors(geometry, unit_positions.detach())
            residual = torch.linalg.vector_norm(inputs.position_gradient.flatten(1), dim=1)
            penetration = self._contact_max_penetration(unit_fused, unit_contact)
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
            contact_energy=loss.contact.detach(),
            contact_max_penetration=penetration,
        )

    # -- physical step bookkeeping ---------------------------------------------------------------------------

    def _cpu_snapshot(self, value: Tensor, name: str) -> Tensor:
        if not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a CPU float32 tensor")
        if value.shape != (len(self._rest.corner_rest_positions), 3) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite [P,3] tensor")
        return value.detach().clone()

    def prepare(self, context_id: str, positions: Tensor, velocities: Tensor, *, forces: Tensor | None = None) -> dict:
        """Snapshot a physical step with one native rigid integration and no energy.

        Inputs and tensor payloads are unbatched detached CPU tensors in SI:
        float32 geometry (positions in meters, velocities in m/s, forces in N)
        and int64 contact indices. The payload contains only a context
        identifier and tensors, with no model, factor or native handle. The
        rigid predictor and the inertial prediction work on the SI data with
        the SI masses and the context's gravity; only the candidate fusion (a
        zero increment of ``target_modes`` modes) runs in cell units through
        the shared factor. Positive pin masses participate in momentum and
        inertia. Prescribed corners keep their input positions in the
        initialized candidate. ``physical_positions`` remains the physical-step
        anchor for every inner update. The rigid initializer never writes
        optimizer history.

        Contact detection (:func:`.contact_scene.detect_contacts`) runs once
        here on the step-start sample positions with the sample velocities
        (mean of the four corner velocities) widening the search band and the
        step-start face normals dropping partners that do not oppose a face,
        so the pair list is frozen for every inner update of the step. The result is
        stored under ``contact_sample_index`` [Q] and ``contact_kind`` [Q]
        (int64), ``contact_partner_point`` [Q, 3], ``contact_partner_normal``
        [Q, 3] and ``contact_partner_radius`` [Q] (float32, SI); ``Q`` is zero
        for a contact-free context or when nothing is near.
        """
        context = self._lookup((context_id,), 1)[0]
        x = self._cpu_snapshot(positions, "positions")
        velocity = self._cpu_snapshot(velocities, "velocities")
        force = torch.zeros_like(x) if forces is None else self._cpu_snapshot(forces, "forces")
        with torch.no_grad():
            corners = self.face_samples.corners
            pairs = detect_contacts(
                sample_points(x[None], corners)[0],
                sample_points(velocity[None], corners)[0],
                context.contact,
                radius=self.contact_radius,
                time_step=self.time_step,
                max_pairs_per_sample=self.contact_max_pairs,
                sample_normals=sample_normals(x[None], corners, self.face_samples.rest_normals)[0],
            )
        with torch.no_grad(), context.lock:
            rigid = context.predictor.predict(x, velocity, force, self.time_step)
            fixed_positions = x[self._fixed_cpu].clone()[None]
            base = x[None] @ rigid.rigid_delta_rotation.transpose(-1, -2) + rigid.rigid_delta_translation[:, None]
            zero_increment = x.new_zeros((1, len(self._rest.cell_corner_indices), 3, self.target_modes))
            unit_base, unit_fixed = self._to_unit(base, fixed_positions)
            candidate = self._from_unit_positions(
                self._fusion.fuse(unit_base, zero_increment, unit_fixed), fixed_positions
            )
            inertial = make_inertial_prediction(
                x[None],
                velocity[None],
                self.time_step,
                explicit_acceleration=context.gravity_cpu + force / context.mass[:, None],
            )[0]
        return {
            "context_id": context_id,
            "physical_positions": x,
            "velocities": velocity,
            "candidate": candidate[0].detach(),
            "inertial_prediction": inertial.detach(),
            "fixed_positions": fixed_positions[0],
            "forces": force,
            "contact_sample_index": pairs.sample_index,
            "contact_kind": pairs.kind,
            "contact_partner_point": pairs.partner_point,
            "contact_partner_normal": pairs.partner_normal,
            "contact_partner_radius": pairs.partner_radius,
        }

    def advance(self, payload: dict) -> dict:
        """Commit candidate displacement to velocity and prepare the next step once.

        Going through :meth:`prepare` refreshes the frozen contact pairs on the
        committed shape exactly once per physical step.
        """
        candidate = self._cpu_snapshot(payload["candidate"], "candidate")
        previous = self._cpu_snapshot(payload["physical_positions"], "physical_positions")
        velocity = (candidate - previous) / self.time_step
        velocity[self._fixed_cpu] = 0
        return self.prepare(payload["context_id"], candidate, velocity, forces=payload.get("forces"))
