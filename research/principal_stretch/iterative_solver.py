# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fixed-work recurrent orchestration for architecture-v5 stretch updates.

One iteration has the auditable order

``residual -> learned delta_H/omega -> compatibility projection -> constraint -> residual``.

The constraint boundary is intentionally outside both the graph model and the
pure compatibility projection.  A future DAT adapter can refresh collision
state in :meth:`IterationConstraintHook.prepare_iteration` and truncate the
projected displacement in :meth:`IterationConstraintHook.constrain`.  This
module provides only the collision-free identity implementation; using a
custom hook is not, by itself, evidence that DAT's no-crossing guarantee holds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from typing import Protocol

import torch

from . import torch_solver
from .predictor import StretchPredictor, predictor_architecture_version
from .torch_solver import ProjectionDiagnostics, SolverState
from .v5_objective import (
    CommonObjectiveContext,
    _common_objective_components_trusted,
    _common_objective_residual_trusted,
)

_PHYSICAL_STEP_TENSOR_FIELDS = (
    "x_current",
    "x_previous",
    "force",
    "gravity",
    "mu",
    "lam",
    "pin",
    "pinned_targets",
)
_MAX_STANDALONE_NONINCREASE_TOLERANCE = 1.0e-6


def _physical_step_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"pr2901-v5-physical-step-context-v1\0")
    for name in _PHYSICAL_STEP_TENSOR_FIELDS:
        value = tensors[name].detach().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = value.view(torch.uint8).cpu().numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class PhysicalStepContext:
    """Owned physical inputs consumed independently by one learned timestep.

    The context owns a clone of every tensor and returns clones from its public
    tensor attributes, so ordinary in-place mutation cannot change the sealed
    inputs. Construction and the checks surrounding every externally supplied
    constraint-hook call copy the canonical bytes to the host for SHA-256;
    that cold per-step authentication cost must be reported separately from a
    captured hot scope. This identity
    is separate from the common objective because different histories and
    loads can produce the same force-shifted inertial target while exposing
    different learned features.

    Args:
        x_current: Physical positions at the beginning of the timestep [m].
        x_previous: Previous physical positions [m].
        force: External nodal forces [N].
        gravity: Gravity [m/s^2].
        mu: Per-tet first material coefficient [Pa].
        lam: Per-tet second material coefficient [Pa].
        pin: Per-tet pin-incidence features.
        pinned_targets: Exact Dirichlet targets [m].
    """

    x_current: torch.Tensor
    x_previous: torch.Tensor
    force: torch.Tensor
    gravity: torch.Tensor
    mu: torch.Tensor
    lam: torch.Tensor
    pin: torch.Tensor
    pinned_targets: torch.Tensor
    physical_step_sha256: str = dataclasses.field(init=False)
    _sealed: bool = dataclasses.field(init=False, repr=False, default=False)

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in _PHYSICAL_STEP_TENSOR_FIELDS and object.__getattribute__(self, "_sealed"):
            return value.clone()
        return value

    def __post_init__(self) -> None:
        for name in _PHYSICAL_STEP_TENSOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.layout != torch.strided:
                raise ValueError(f"{name} must have strided layout")
            if not value.is_floating_point():
                raise ValueError(f"{name} must have a floating dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")

        if self.x_current.ndim not in (2, 3) or self.x_current.shape[-1] != 3:
            raise ValueError("x_current must have shape (V, 3) or (B, V, 3)")
        for name in ("x_previous", "force"):
            value = getattr(self, name)
            if value.shape != self.x_current.shape:
                raise ValueError(f"{name} must have the same shape as x_current")
        expected_gravity_shapes = ((3,), (*self.x_current.shape[:-2], 3))
        if self.gravity.shape not in expected_gravity_shapes:
            raise ValueError("gravity must have shape (3,) or match the position batch dimensions")
        if self.mu.ndim not in (1, 2) or self.lam.shape != self.mu.shape or self.pin.shape != self.mu.shape:
            raise ValueError("mu, lam, and pin must share a one- or two-dimensional shape")
        if self.x_current.ndim == 2 and self.mu.ndim != 1:
            raise ValueError("unbatched positions require unbatched material features")
        if self.x_current.ndim == 3 and self.mu.ndim == 2 and self.mu.shape[0] != self.x_current.shape[0]:
            raise ValueError("batched material features must match the position batch")
        if self.pinned_targets.ndim not in (2, 3) or self.pinned_targets.shape[-1] != 3:
            raise ValueError("pinned_targets must have shape (P, 3) or (B, P, 3)")

        device = self.x_current.device
        position_dtype = self.x_current.dtype
        for name in ("x_previous", "force", "gravity", "mu", "lam", "pin", "pinned_targets"):
            value = getattr(self, name)
            if value.device != device:
                raise ValueError("all physical-step tensors must share one device")
        for name in ("x_previous", "force", "gravity", "mu", "lam", "pin", "pinned_targets"):
            if getattr(self, name).dtype != position_dtype:
                raise ValueError(f"{name} must share the position dtype")
        if not torch.logical_or(self.pin == 0.0, self.pin == 1.0).all():
            raise ValueError("pin features must contain only exact zero or one values")

        for name in _PHYSICAL_STEP_TENSOR_FIELDS:
            object.__setattr__(self, name, getattr(self, name).clone())
        tensors = {name: getattr(self, name) for name in _PHYSICAL_STEP_TENSOR_FIELDS}
        object.__setattr__(self, "physical_step_sha256", _physical_step_digest(tensors))
        object.__setattr__(self, "_sealed", True)

    def validate_immutable(self) -> None:
        """Reauthenticate the sealed context against its canonical bytes."""
        if not self._sealed:
            raise RuntimeError("physical-step context is not sealed")
        tensors = {name: object.__getattribute__(self, name) for name in _PHYSICAL_STEP_TENSOR_FIELDS}
        if _physical_step_digest(tensors) != self.physical_step_sha256:
            raise RuntimeError("physical-step context changed after authentication")

    def _validate_sealed(self) -> None:
        """Check construction state without repeating cold byte authentication."""
        if not self._sealed:
            raise RuntimeError("physical-step context is not sealed")

    def _owned_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return zero-copy owned tensors for internal solver code."""
        return tuple(object.__getattribute__(self, name) for name in _PHYSICAL_STEP_TENSOR_FIELDS)


@dataclasses.dataclass(frozen=True)
class IterativeSolverConfig:
    """Fixed learned work and fail-closed state checks.

    Args:
        iterations: Number of weight-shared learned iterations.
        detach_residual_features: Detach the exact analytic residual before it
            enters the graph network. This avoids a Hessian-through-residual
            training path while positions still differentiate through every
            learned update and compatibility projection.
        minimum_determinant: Strict lower bound for every committed tet
            determinant. The default requires positive orientation.
        minimum_singular_value: Strict lower bound for every committed
            deformation-gradient singular value.
        objective_policy: ``"record"`` records the exact common objective;
            ``"require-nonincreasing"`` rejects an increasing iteration.
        residual_policy: ``"record"`` records the normalized free-residual
            norm; ``"require-nonincreasing"`` rejects an increasing iteration.
        objective_increase_tolerance: Allowed absolute common-objective
            increase [J] under the nonincrease policy.
        normalized_residual_increase_tolerance: Allowed absolute dimensionless
            residual-norm increase under the nonincrease policy.
        initializer_policy: Authenticated initial-iterate policy. The current
            implementation admits only physical-state persistence.
        return_projection_diagnostics: Request per-iteration compatibility
            work diagnostics. This performs host-visible scalar extraction in
            the current projection implementation and is therefore disabled
            in differentiable/hot execution by default.
        head_mode: ``"learned"`` for the ordinary learned route, or a work-controlled
            ``"zero"``/``"permuted"`` learned-contribution ablation.
        head_permutation: Explicit all-tet permutation for the permuted-head
            ablation. The model is still executed once per iteration.
    """

    iterations: int
    detach_residual_features: bool = True
    minimum_determinant: float = 0.0
    minimum_singular_value: float = 0.0
    objective_policy: str = "require-nonincreasing"
    residual_policy: str = "require-nonincreasing"
    objective_increase_tolerance: float = 1.0e-12
    normalized_residual_increase_tolerance: float = 1.0e-12
    initializer_policy: str = "persistence"
    return_projection_diagnostics: bool = True
    head_mode: str = "learned"
    head_permutation: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, numbers.Integral):
            raise TypeError("iterations must be an integer")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not isinstance(self.detach_residual_features, bool):
            raise TypeError("detach_residual_features must be a bool")
        if not isinstance(self.return_projection_diagnostics, bool):
            raise TypeError("return_projection_diagnostics must be a bool")
        if isinstance(self.minimum_determinant, bool) or not isinstance(self.minimum_determinant, numbers.Real):
            raise TypeError("minimum_determinant must be a real number")
        if not math.isfinite(self.minimum_determinant) or self.minimum_determinant < 0.0:
            raise ValueError("minimum_determinant must be finite and non-negative")
        for name in (
            "minimum_singular_value",
            "objective_increase_tolerance",
            "normalized_residual_increase_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.objective_increase_tolerance > _MAX_STANDALONE_NONINCREASE_TOLERANCE:
            raise ValueError("objective_increase_tolerance must remain a registered small safeguard tolerance")
        if self.normalized_residual_increase_tolerance > _MAX_STANDALONE_NONINCREASE_TOLERANCE:
            raise ValueError(
                "normalized_residual_increase_tolerance must remain a registered small safeguard tolerance"
            )
        if self.objective_policy not in ("record", "require-nonincreasing"):
            raise ValueError("objective_policy must be 'record' or 'require-nonincreasing'")
        if self.residual_policy not in ("record", "require-nonincreasing"):
            raise ValueError("residual_policy must be 'record' or 'require-nonincreasing'")
        if self.initializer_policy != "persistence":
            raise ValueError("initializer_policy must be 'persistence'")
        if self.head_mode not in ("learned", "zero", "permuted"):
            raise ValueError("head_mode must be 'learned', 'zero', or 'permuted'")
        if self.head_mode == "permuted":
            if not isinstance(self.head_permutation, tuple) or not self.head_permutation:
                raise ValueError("permuted head mode requires an explicit non-empty tuple")
            if any(
                isinstance(index, bool) or not isinstance(index, numbers.Integral) for index in self.head_permutation
            ):
                raise TypeError("head_permutation entries must be integers")
        elif self.head_permutation is not None:
            raise ValueError("head_permutation is only valid with head_mode='permuted'")


@dataclasses.dataclass(frozen=True)
class ConstraintObservation:
    """Constraint state and invariant features prepared before one learned pass."""

    state: object
    normal: torch.Tensor
    normalized_slack: torch.Tensor
    diagnostics: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class ConstraintApplication:
    """Committed constrained positions and updated constraint state."""

    state: object
    positions: torch.Tensor
    diagnostics: Mapping[str, object]


class IterationConstraintHook(Protocol):
    """Stateful displacement-constraint boundary for one implicit step."""

    def descriptor(self) -> dict[str, object]:
        """Return immutable JSON-compatible constraint semantics."""

    def begin_step(
        self,
        positions: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> object:
        """Create step-local constraint state from the initial iterate."""

    def prepare_iteration(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
    ) -> ConstraintObservation:
        """Refresh constraint data and expose current normal/slack features."""

    def constrain(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
        proposed: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> ConstraintApplication:
        """Constrain the proposed displacement before it is committed."""


def _expand_pinned_targets(
    positions: torch.Tensor,
    pinned: torch.Tensor,
    pinned_targets: torch.Tensor,
) -> torch.Tensor:
    expected_tail = (int(pinned.numel()), 3)
    if pinned_targets.shape[-2:] != expected_tail:
        raise ValueError(f"pinned_targets must end in {expected_tail}, got {tuple(pinned_targets.shape)}")
    try:
        batch = torch.broadcast_shapes(positions.shape[:-2], pinned_targets.shape[:-2])
    except RuntimeError as error:
        raise ValueError("pinned_targets batch dimensions are not broadcastable to positions") from error
    if batch != positions.shape[:-2]:
        raise ValueError("pinned_targets must not introduce new position batch dimensions")
    return pinned_targets.expand(*batch, *expected_tail)


class IdentityConstraintHook:
    """Collision-free hook that commits the projected proposal unchanged."""

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "identity",
            "refresh_policy": "none",
            "displacement_reference": "current-iterate",
        }

    def begin_step(
        self,
        positions: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> object:
        del pinned, pinned_targets
        return {"initial_positions": positions}

    def prepare_iteration(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
    ) -> ConstraintObservation:
        del iteration
        return ConstraintObservation(
            state=state,
            normal=torch.zeros_like(positions),
            normalized_slack=torch.zeros(positions.shape[:-1], dtype=positions.dtype, device=positions.device),
            diagnostics={"refreshes": 0},
        )

    def constrain(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
        proposed: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> ConstraintApplication:
        del iteration, positions
        targets = _expand_pinned_targets(proposed, pinned, pinned_targets)
        committed = proposed.index_copy(-2, pinned, targets)
        return ConstraintApplication(
            state=state,
            positions=committed,
            diagnostics={"truncation_calls": 0, "minimum_fraction": 1.0},
        )


@dataclasses.dataclass(frozen=True)
class IterativeSolverWork:
    """Exact top-level work counts for one recurrent solve."""

    predictor_passes: int
    projection_calls: int
    residual_evaluations: int
    objective_evaluations: int
    state_validity_evaluations: int
    constraint_preparations: int
    constraint_applications: int
    physical_step_authentications: int
    common_objective_authentications: int
    projection_backend: str
    projection_diagnostics_recorded: bool
    projection_iterations: int | None
    projection_matrix_vector_products: int | None
    projection_preconditioner_applications: int | None
    projection_factor_solves: int | None


@dataclasses.dataclass
class IterativeSolverIteration:
    """Differentiable tensors and diagnostics for one accepted learned iteration."""

    iteration: int
    iteration_fraction: float
    positions_before: torch.Tensor
    normalized_residual_before: torch.Tensor
    raw_residual_norm_before: torch.Tensor
    normalized_residual_norm_before: torch.Tensor
    objective_before: torch.Tensor
    minimum_determinant_before: torch.Tensor
    minimum_singular_value_before: torch.Tensor
    delta_h: torch.Tensor
    omega: torch.Tensor
    target_deformation_gradient: torch.Tensor
    proposed_positions: torch.Tensor
    positions: torch.Tensor
    residual_after: torch.Tensor
    raw_residual_norm_after: torch.Tensor
    normalized_residual_norm_after: torch.Tensor
    objective_after: torch.Tensor
    minimum_determinant_after: torch.Tensor
    minimum_singular_value_after: torch.Tensor
    projection_diagnostics: ProjectionDiagnostics | None
    constraint_prepare_diagnostics: Mapping[str, object]
    constraint_diagnostics: Mapping[str, object]


@dataclasses.dataclass
class IterativeSolverResult:
    """Final constrained iterate, residual, work, and full learned trace.

    ``projection_state_sha256`` and ``static_graph_sha256`` identify the
    solver inputs authenticated at entry. They are not, by themselves, proof
    that arbitrary Python constraint code left those inputs unchanged while
    the solver ran. ``constraint_registration`` makes that claim boundary
    explicit: only the built-in identity hook is registered in this
    foundation, and authenticated replay is provided separately by the v5
    checkpoint verifier. Custom-hook results remain development diagnostics
    until that hook has its own registered implementation and replay contract.
    """

    positions: torch.Tensor
    normalized_residual: torch.Tensor
    trace: tuple[IterativeSolverIteration, ...]
    work: IterativeSolverWork
    constraint_descriptor: dict[str, object]
    constraint_descriptor_sha256: str
    constraint_registration: str
    head_mode: str
    head_permutation: tuple[int, ...] | None
    physical_step_sha256: str
    common_objective_sha256: str
    projection_state_sha256: str
    static_graph_sha256: str
    objective: torch.Tensor
    raw_residual_norm: torch.Tensor
    normalized_residual_norm: torch.Tensor
    minimum_determinant: torch.Tensor
    minimum_singular_value: torch.Tensor


def _canonical_constraint_descriptor(hook: IterationConstraintHook) -> tuple[dict[str, object], str]:
    descriptor = hook.descriptor()
    if not isinstance(descriptor, dict):
        raise ValueError("constraint descriptor must be a dictionary")
    if descriptor.get("schema_version") != 1 or not isinstance(descriptor.get("kind"), str):
        raise ValueError("constraint descriptor must contain schema_version=1 and a string kind")
    try:
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("constraint descriptor must be finite JSON data") from error
    return canonical, hashlib.sha256(encoded).hexdigest()


def _validate_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains a non-finite value")


def _validate_committed_state(
    projection_state: SolverState,
    positions: torch.Tensor,
    pinned_targets: torch.Tensor,
    minimum_determinant: float,
    minimum_singular_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_finite("committed positions", positions)
    targets = _expand_pinned_targets(positions, projection_state.pinned, pinned_targets)
    if not torch.equal(positions[..., projection_state.pinned, :], targets):
        raise RuntimeError("constraint hook changed an exact pinned target")
    deformation_gradient = torch_solver.compute_F(positions, projection_state.tets, projection_state.J)
    determinant = torch.linalg.det(deformation_gradient)
    if not torch.isfinite(determinant).all():
        raise RuntimeError("committed deformation determinant is non-finite")
    if (determinant <= minimum_determinant).any():
        minimum = float(determinant.detach().min())
        raise RuntimeError(
            f"committed state violates determinant bound: minimum {minimum:.6e} <= {minimum_determinant:.6e}"
        )
    singular_values = torch.linalg.svdvals(deformation_gradient)
    if not torch.isfinite(singular_values).all():
        raise RuntimeError("committed deformation singular values are non-finite")
    if (singular_values <= minimum_singular_value).any():
        minimum = float(singular_values.detach().min())
        raise RuntimeError(
            f"committed state violates singular-value bound: minimum {minimum:.6e} <= {minimum_singular_value:.6e}"
        )
    return determinant.amin(dim=-1), singular_values.amin(dim=(-2, -1))


def _validate_problem_identity(
    predictor: StretchPredictor,
    projection_state: SolverState,
    objective: CommonObjectiveContext,
) -> None:
    objective.validate_immutable()
    actual_projection_sha256 = torch_solver.projection_state_sha256(projection_state)
    if projection_state.projection_state_sha256 != actual_projection_sha256:
        raise ValueError("compatibility projection state differs from its authenticated identity")
    if predictor.kind != "graph-transformer":
        raise ValueError("iterative v5 requires a graph-transformer predictor")
    model = predictor.model
    if getattr(model, "static_graph_sha256", None) != model.compute_static_graph_sha256():
        raise ValueError("predictor static graph differs from its authenticated identity")
    if getattr(model, "static_mesh_sha256", None) != projection_state.static_mesh_sha256:
        raise ValueError("predictor and projection static-mesh identities differ")
    if model.tets.device != projection_state.tets.device or not torch.equal(model.tets, projection_state.tets):
        raise ValueError("predictor and projection ordered tetrahedra differ")
    if projection_state.n_verts != objective.n_vertices or projection_state.n_tets != objective.n_tets:
        raise ValueError("projection and common-objective mesh sizes differ")
    if projection_state.rest_q.device != objective.device or projection_state.rest_q.dtype != objective.dtype:
        raise ValueError("projection and common objective must share device and dtype")
    for name, projected, common in (
        ("tets", projection_state.tets, objective.tets),
        ("J", projection_state.J, objective.J),
        ("volume", projection_state.w, objective.volume),
        ("pinned", projection_state.pinned, objective.pinned),
    ):
        if not torch.equal(projected, common):
            raise ValueError(f"projection and common-objective {name} differ")


def _validate_physical_context(
    projection_state: SolverState,
    objective: CommonObjectiveContext,
    physical_step: PhysicalStepContext,
) -> None:
    physical_step._validate_sealed()
    x_current, x_previous, force, gravity, mu, lam, pin, _pinned_targets = physical_step._owned_tensors()
    for name, model_value in (("mu", mu), ("lam", lam)):
        objective_value = objective._owned_tensor(name)
        expected = objective_value.expand_as(model_value)
        if not torch.equal(model_value, expected):
            raise ValueError(f"model {name} features differ from the bound common objective")

    expected_pin = torch.isin(projection_state.tets, projection_state.pinned).any(dim=-1).to(pin)
    expected_pin = expected_pin.expand_as(pin)
    if not torch.equal(pin, expected_pin):
        raise ValueError("model pin-incidence features differ from the projection constraints")

    free_mask = torch.ones(projection_state.n_verts, dtype=torch.bool, device=objective.device)
    free_mask[projection_state.pinned] = False
    mass = objective._owned_tensor("mass")
    inertial_target = objective._owned_tensor("inertial_target")
    acceleration = gravity[..., None, :] + force[..., free_mask, :] / mass[free_mask, None]
    expected_target = 2.0 * x_current[..., free_mask, :] - x_previous[..., free_mask, :]
    expected_target = expected_target + objective.dt * objective.dt * acceleration
    bound_target = inertial_target[free_mask].expand_as(expected_target)
    if not torch.allclose(expected_target, bound_target, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("physical history, loads, and timestep differ from the bound inertial target")


def _normalized_residual_norm(residual: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(residual.flatten(start_dim=-2), dim=-1)


def _reauthenticate_contexts(
    physical_step: PhysicalStepContext,
    objective: CommonObjectiveContext,
) -> None:
    """Check canonical bytes after one externally supplied hook call."""
    physical_step.validate_immutable()
    objective.validate_immutable()


def _validate_config_execution_dtype(config: IterativeSolverConfig, reference: torch.Tensor) -> None:
    """Reject scalar safeguards that change meaning in the execution dtype."""
    for name in (
        "minimum_determinant",
        "minimum_singular_value",
        "objective_increase_tolerance",
        "normalized_residual_increase_tolerance",
    ):
        python_value = getattr(config, name)
        materialized = reference.new_tensor(python_value)
        if not bool(torch.isfinite(materialized).item()):
            raise ValueError(f"{name} must remain finite in execution dtype {reference.dtype}")
        if python_value > 0.0 and not bool((materialized > 0.0).item()):
            raise ValueError(f"{name} must remain positive in execution dtype {reference.dtype}")


def _enforce_nonincrease(
    name: str,
    policy: str,
    before: torch.Tensor,
    after: torch.Tensor,
    tolerance: float,
) -> None:
    if policy == "require-nonincreasing" and (after > before + tolerance).any():
        increase = float((after - before).detach().max())
        raise RuntimeError(f"committed iteration increased {name} by {increase:.6e}")


def solve_iterative_principal_stretch(
    *,
    predictor: StretchPredictor,
    projection_state: SolverState,
    objective: CommonObjectiveContext,
    physical_step: PhysicalStepContext,
    expected_physical_step_sha256: str,
    config: IterativeSolverConfig,
    constraint: IterationConstraintHook | None = None,
) -> IterativeSolverResult:
    """Run a fixed number of weight-shared learned principal-stretch iterations.

    The exact contact-free common residual is a learned feature and evaluation
    quantity, not a classical correction step. The network is called once in
    every iteration and its bounded ``delta_H``/``omega`` field is always
    decoded before the constraint hook. A constraint that returns the previous
    iterate is an explicit zero-motion event; this function never substitutes
    an inertial, VBD, or Newton endpoint.

    The runtime consistency check proves that the supplied physical history and
    loads reproduce the common objective's free-vertex inertial target; it does
    not replace dataset-level authentication of those tensors individually.
    Loads on eliminated pinned rows are zeroed before learned feature assembly
    because they do no work in the bound objective.

    A custom Python constraint is an intentionally open research seam. Its
    descriptor and calls are recorded, but this function does not sandbox the
    callback or authenticate its execution. Only a separately registered hook
    plus independent runtime replay may be used as solver evidence; currently
    that stricter path admits only :class:`IdentityConstraintHook`.

    Args:
        predictor: Architecture-v5 graph predictor shared across iterations.
        projection_state: Mesh-static compatibility projection state.
        objective: Exact device-resident common-objective context.
        physical_step: Authenticated physical history and learned inputs.
        expected_physical_step_sha256: Canonical physical-step identity supplied
            by the verified sample/evaluation binding.
        config: Fixed learned work and validity policy.
        constraint: Post-projection displacement constraint. Defaults to the
            collision-free identity hook.

    Returns:
        Final state, normalized residual, differentiable per-iteration trace,
        exact work counts, and the constraint descriptor.
    """
    if not isinstance(config, IterativeSolverConfig):
        raise TypeError("config must be an IterativeSolverConfig")
    if predictor_architecture_version(predictor) != 5:
        raise ValueError("iterative principal-stretch solving requires an architecture-v5 predictor")
    _validate_problem_identity(predictor, projection_state, objective)
    if not isinstance(physical_step, PhysicalStepContext):
        raise TypeError("physical_step must be a PhysicalStepContext")
    if (
        not isinstance(expected_physical_step_sha256, str)
        or len(expected_physical_step_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_physical_step_sha256)
    ):
        raise ValueError("expected_physical_step_sha256 must be a lowercase SHA-256 digest")
    physical_step.validate_immutable()
    if physical_step.physical_step_sha256 != expected_physical_step_sha256:
        raise ValueError("physical-step identity differs from the verified sample binding")
    x_current, x_previous, force, gravity, mu, lam, pin, pinned_targets = physical_step._owned_tensors()
    _validate_config_execution_dtype(config, x_current)
    x_initial = x_current
    if x_initial.ndim not in (2, 3):
        raise ValueError("the graph predictor supports unbatched (V, 3) or batched (B, V, 3) positions")
    expected_positions = (projection_state.n_verts, 3)
    for name, value in (
        ("x_current", x_current),
        ("x_previous", x_previous),
        ("x_initial", x_initial),
        ("force", force),
    ):
        if value.shape[-2:] != expected_positions:
            raise ValueError(f"{name} must end in {expected_positions}, got {tuple(value.shape)}")
        if value.shape != x_initial.shape:
            raise ValueError("position, history, and force tensors must have identical shapes")
        if value.device != objective.device or value.dtype != objective.dtype:
            raise ValueError(f"{name} must share the common objective's device and dtype")
        _validate_finite(name, value)
    if gravity.shape not in ((3,), (*x_initial.shape[:-2], 3)):
        raise ValueError("gravity must have shape (3,) or match the position batch dimensions")
    if gravity.device != objective.device:
        raise ValueError("gravity must share the common objective's device")
    _validate_finite("gravity", gravity)
    expected_material_shapes = ((projection_state.n_tets,),)
    if x_initial.ndim == 3:
        expected_material_shapes += ((x_initial.shape[0], projection_state.n_tets),)
    if mu.shape not in expected_material_shapes or lam.shape != mu.shape or pin.shape != mu.shape:
        raise ValueError("mu, lam, and pin must share an allowed unbatched or batched tet shape")
    for name, value in (("mu", mu), ("lam", lam), ("pin", pin)):
        if value.device != objective.device:
            raise ValueError(f"{name} must share the common objective's device")
        _validate_finite(name, value)
    _validate_physical_context(projection_state, objective, physical_step)
    targets = _expand_pinned_targets(x_initial, projection_state.pinned, pinned_targets)
    if not torch.equal(x_initial[..., projection_state.pinned, :], targets):
        raise ValueError("x_initial must contain the exact pinned targets")

    hook: IterationConstraintHook = IdentityConstraintHook() if constraint is None else constraint
    descriptor, descriptor_sha256 = _canonical_constraint_descriptor(hook)
    constraint_registration = (
        "registered-identity-development"
        if type(hook) is IdentityConstraintHook
        else "unregistered-custom-no-authenticated-execution"
    )

    current = x_initial
    minimum_determinant, minimum_singular_value = _validate_committed_state(
        projection_state,
        current,
        pinned_targets,
        config.minimum_determinant,
        config.minimum_singular_value,
    )
    constraint_state = hook.begin_step(current, projection_state.pinned, pinned_targets)
    _reauthenticate_contexts(physical_step, objective)
    raw_residual = _common_objective_residual_trusted(
        objective,
        current,
        detach=config.detach_residual_features,
    )
    residual = raw_residual / objective.residual_scale
    _validate_finite("initial common residual", residual)
    raw_residual_norm = _normalized_residual_norm(raw_residual)
    normalized_residual_norm = _normalized_residual_norm(residual)
    objective_value = _common_objective_components_trusted(objective, current)["total"]
    _validate_finite("initial common objective", objective_value)
    trace: list[IterativeSolverIteration] = []
    # Loads on eliminated Dirichlet rows do no work in the common objective.
    # Mask them before feature construction so an objective-null pinned load
    # cannot alter the learned prediction.
    network_force = force.clone()
    network_force[..., projection_state.pinned, :] = 0.0
    head_permutation = None
    if config.head_permutation is not None:
        head_permutation = torch.tensor(config.head_permutation, dtype=torch.int64, device=current.device)

    for iteration in range(config.iterations):
        iteration_fraction = iteration / max(config.iterations - 1, 1)
        observation = hook.prepare_iteration(constraint_state, iteration, current)
        _reauthenticate_contexts(physical_step, objective)
        if not isinstance(observation, ConstraintObservation):
            raise RuntimeError("constraint hook prepare_iteration must return ConstraintObservation")
        if observation.normal.shape != current.shape:
            raise RuntimeError("constraint normal must have the same shape as positions")
        if observation.normalized_slack.shape != current.shape[:-1]:
            raise RuntimeError("constraint slack must match positions without the vector dimension")
        if observation.normal.device != current.device or observation.normal.dtype != current.dtype:
            raise RuntimeError("constraint normal must share the position device and dtype")
        if observation.normalized_slack.device != current.device or observation.normalized_slack.dtype != current.dtype:
            raise RuntimeError("constraint slack must share the position device and dtype")
        _validate_finite("constraint normal", observation.normal)
        _validate_finite("constraint slack", observation.normalized_slack)
        constraint_state = observation.state

        target_f, delta_h, omega = predictor.predict_principal_stretch_update(
            projection_state,
            x_current,
            x_previous,
            current,
            network_force,
            gravity,
            mu,
            lam,
            pin,
            residual,
            observation.normal,
            observation.normalized_slack,
            iteration_fraction=iteration_fraction,
            physical_dt=objective.dt,
            head_mode=config.head_mode,
            head_permutation=head_permutation,
        )
        _validate_finite("learned target deformation gradient", target_f)
        projection_kwargs: dict[str, object] = {}
        if projection_state.projection_backend != "dense":
            projection_kwargs["initial_positions"] = current
        projection_diagnostics: ProjectionDiagnostics | None = None
        if config.return_projection_diagnostics:
            proposed, projection_diagnostics = torch_solver.project_deformation_gradient(
                projection_state,
                target_f,
                pinned_targets,
                return_diagnostics=True,
                **projection_kwargs,
            )
        else:
            proposed = torch_solver.project_deformation_gradient(
                projection_state,
                target_f,
                pinned_targets,
                **projection_kwargs,
            )
        _validate_finite("projected proposal", proposed)

        application = hook.constrain(
            constraint_state,
            iteration,
            current,
            proposed,
            projection_state.pinned,
            pinned_targets,
        )
        _reauthenticate_contexts(physical_step, objective)
        if not isinstance(application, ConstraintApplication):
            raise RuntimeError("constraint hook constrain must return ConstraintApplication")
        constraint_state = application.state
        committed = application.positions
        if committed.shape != current.shape or committed.device != current.device or committed.dtype != current.dtype:
            raise RuntimeError("constraint hook changed the position shape, device, or dtype")
        minimum_determinant_after, minimum_singular_value_after = _validate_committed_state(
            projection_state,
            committed,
            pinned_targets,
            config.minimum_determinant,
            config.minimum_singular_value,
        )
        raw_residual_after = _common_objective_residual_trusted(
            objective,
            committed,
            detach=config.detach_residual_features,
        )
        residual_after = raw_residual_after / objective.residual_scale
        _validate_finite("committed common residual", residual_after)
        raw_residual_norm_after = _normalized_residual_norm(raw_residual_after)
        normalized_residual_norm_after = _normalized_residual_norm(residual_after)
        objective_after = _common_objective_components_trusted(objective, committed)["total"]
        _validate_finite("committed common objective", objective_after)
        _enforce_nonincrease(
            "common objective",
            config.objective_policy,
            objective_value,
            objective_after,
            config.objective_increase_tolerance,
        )
        _enforce_nonincrease(
            "normalized residual norm",
            config.residual_policy,
            normalized_residual_norm,
            normalized_residual_norm_after,
            config.normalized_residual_increase_tolerance,
        )
        trace.append(
            IterativeSolverIteration(
                iteration=iteration,
                iteration_fraction=iteration_fraction,
                positions_before=current,
                normalized_residual_before=residual,
                raw_residual_norm_before=raw_residual_norm,
                normalized_residual_norm_before=normalized_residual_norm,
                objective_before=objective_value,
                minimum_determinant_before=minimum_determinant,
                minimum_singular_value_before=minimum_singular_value,
                delta_h=delta_h,
                omega=omega,
                target_deformation_gradient=target_f,
                proposed_positions=proposed,
                positions=committed,
                residual_after=residual_after,
                raw_residual_norm_after=raw_residual_norm_after,
                normalized_residual_norm_after=normalized_residual_norm_after,
                objective_after=objective_after,
                minimum_determinant_after=minimum_determinant_after,
                minimum_singular_value_after=minimum_singular_value_after,
                projection_diagnostics=projection_diagnostics,
                constraint_prepare_diagnostics=dict(observation.diagnostics),
                constraint_diagnostics=dict(application.diagnostics),
            )
        )
        current = committed
        residual = residual_after
        raw_residual_norm = raw_residual_norm_after
        normalized_residual_norm = normalized_residual_norm_after
        objective_value = objective_after
        minimum_determinant = minimum_determinant_after
        minimum_singular_value = minimum_singular_value_after

    projection_records = tuple(item.projection_diagnostics for item in trace)
    projection_diagnostics_recorded = all(item is not None for item in projection_records)
    if projection_diagnostics_recorded:
        recorded = tuple(item for item in projection_records if item is not None)
        projection_iterations = sum(item.iterations for item in recorded)
        projection_matrix_vector_products = sum(item.matrix_vector_products for item in recorded)
        projection_preconditioner_applications = sum(item.preconditioner_applications for item in recorded)
        projection_factor_solves = sum(item.factor_solves for item in recorded)
    else:
        projection_iterations = None
        projection_matrix_vector_products = None
        projection_preconditioner_applications = None
        projection_factor_solves = None

    # Reauthenticate after all user-supplied hook calls so a hook retaining an
    # internal context reference cannot silently change the bound problem
    # during execution. These canonical byte checks are explicitly cold
    # per-step evidence work, not hidden inside every residual evaluation.
    physical_step.validate_immutable()
    objective.validate_immutable()

    return IterativeSolverResult(
        positions=current,
        normalized_residual=residual,
        trace=tuple(trace),
        work=IterativeSolverWork(
            predictor_passes=config.iterations,
            projection_calls=config.iterations,
            residual_evaluations=config.iterations + 1,
            objective_evaluations=config.iterations + 1,
            state_validity_evaluations=config.iterations + 1,
            constraint_preparations=config.iterations,
            constraint_applications=config.iterations,
            physical_step_authentications=2 * config.iterations + 3,
            common_objective_authentications=2 * config.iterations + 3,
            projection_backend=projection_state.projection_backend,
            projection_diagnostics_recorded=projection_diagnostics_recorded,
            projection_iterations=projection_iterations,
            projection_matrix_vector_products=projection_matrix_vector_products,
            projection_preconditioner_applications=projection_preconditioner_applications,
            projection_factor_solves=projection_factor_solves,
        ),
        constraint_descriptor=descriptor,
        constraint_descriptor_sha256=descriptor_sha256,
        constraint_registration=constraint_registration,
        head_mode=config.head_mode,
        head_permutation=config.head_permutation,
        physical_step_sha256=physical_step.physical_step_sha256,
        common_objective_sha256=objective.common_objective_sha256,
        projection_state_sha256=projection_state.projection_state_sha256,
        static_graph_sha256=predictor.model.static_graph_sha256,
        objective=objective_value,
        raw_residual_norm=raw_residual_norm,
        normalized_residual_norm=normalized_residual_norm,
        minimum_determinant=minimum_determinant,
        minimum_singular_value=minimum_singular_value,
    )


__all__ = [
    "ConstraintApplication",
    "ConstraintObservation",
    "IdentityConstraintHook",
    "IterationConstraintHook",
    "IterativeSolverConfig",
    "IterativeSolverIteration",
    "IterativeSolverResult",
    "IterativeSolverWork",
    "PhysicalStepContext",
    "solve_iterative_principal_stretch",
]
