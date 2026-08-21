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

import numpy as np
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
_SOURCE_INTEGRATION_TENSOR_FIELDS = (
    "pre_event_positions",
    "velocity",
    "mass",
    "inverse_mass",
)
PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64 = "algebraic-float64-position-history-loads-v1"
PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32 = "solver-vbd-staged-float32-v1"
_PHYSICAL_INTEGRATION_POLICIES = (
    PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64,
    PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
)
_MAX_STANDALONE_NONINCREASE_TOLERANCE = 1.0e-6


def _is_canonical_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _update_tensor_digest(digest: object, name: str, value: torch.Tensor) -> None:
    value = value.detach().contiguous()
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


def _array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _source_integration_digest(
    source_transition_sha256: str,
    dt_float32_bits: str,
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256(b"pr2901-v5-solver-vbd-staged-float32-evidence-v1\0")
    metadata = json.dumps(
        {
            "policy": PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            "source_transition_sha256": source_transition_sha256,
            "dt_float32_bits": dt_float32_bits,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    for name in _SOURCE_INTEGRATION_TENSOR_FIELDS:
        _update_tensor_digest(digest, name, tensors[name])
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class SolverVBDStagedFloat32Evidence:
    """Sealed source arithmetic needed to authenticate one PR/VBD step.

    These tensors are authentication-only. They never enter the learned model.
    The exact float32 source order is replayed on CPU before a solve or training
    sample is accepted, avoiding device-specific fusion or rounding ambiguity.

    Args:
        source_transition_sha256: Exact PR transition identity.
        dt_seconds: Source timestep [s], exactly representable as float32.
        pre_event_positions: Committed pre-callback positions [m].
        velocity: Committed pre-callback velocity [m/s].
        mass: SolverVBD particle masses [kg].
        inverse_mass: SolverVBD particle inverse masses [1/kg].
    """

    source_transition_sha256: str
    dt_seconds: float
    pre_event_positions: torch.Tensor
    velocity: torch.Tensor
    mass: torch.Tensor
    inverse_mass: torch.Tensor
    dt_float32_bits: str = dataclasses.field(init=False)
    evidence_sha256: str = dataclasses.field(init=False)
    _sealed: bool = dataclasses.field(init=False, repr=False, default=False)

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in _SOURCE_INTEGRATION_TENSOR_FIELDS and object.__getattribute__(self, "_sealed"):
            return value.clone()
        return value

    def __post_init__(self) -> None:
        source_sha = self.source_transition_sha256
        if not _is_canonical_sha256(source_sha):
            raise ValueError("source_transition_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.dt_seconds, bool) or not isinstance(self.dt_seconds, numbers.Real):
            raise TypeError("source dt_seconds must be a real number")
        dt32 = np.float32(self.dt_seconds)
        if not np.isfinite(dt32) or dt32 <= np.float32(0.0) or float(dt32) != float(self.dt_seconds):
            raise ValueError("source dt_seconds must be finite, positive, and exactly representable as float32")
        dt_bits = f"0x{np.asarray(dt32).view(np.uint32).item():08x}"
        object.__setattr__(self, "dt_seconds", float(dt32))
        object.__setattr__(self, "dt_float32_bits", dt_bits)

        for name in _SOURCE_INTEGRATION_TENSOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"source {name} must be a torch.Tensor")
            if value.layout != torch.strided or value.dtype != torch.float32:
                raise ValueError(f"source {name} must be a strided float32 tensor")
            if value.requires_grad:
                raise ValueError(f"source {name} must not require gradients")
            if not torch.isfinite(value).all():
                raise ValueError(f"source {name} must be finite")
        positions = self.pre_event_positions
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("source pre_event_positions must have shape (V, 3)")
        if self.velocity.shape != positions.shape:
            raise ValueError("source velocity must match pre_event_positions")
        if self.mass.shape != (positions.shape[0],) or self.inverse_mass.shape != self.mass.shape:
            raise ValueError("source mass and inverse_mass must have shape (V,)")
        device = positions.device
        if any(getattr(self, name).device != device for name in _SOURCE_INTEGRATION_TENSOR_FIELDS):
            raise ValueError("all source integration tensors must share one device")
        if (self.mass < 0.0).any() or (self.inverse_mass < 0.0).any():
            raise ValueError("source mass and inverse_mass must be non-negative")
        mass32 = self.mass.detach().contiguous().cpu().numpy().copy()
        inverse_mass32 = self.inverse_mass.detach().contiguous().cpu().numpy().copy()
        expected_inverse_mass = np.zeros_like(mass32)
        positive_mass = mass32 > np.float32(0.0)
        expected_inverse_mass[positive_mass] = (np.float32(1.0) / mass32[positive_mass]).astype(np.float32)
        if not _array_bytes_equal(inverse_mass32, expected_inverse_mass):
            raise ValueError("source inverse_mass must be the exact float32 reciprocal of source mass")

        for name in _SOURCE_INTEGRATION_TENSOR_FIELDS:
            object.__setattr__(self, name, getattr(self, name).clone())
        tensors = {name: object.__getattribute__(self, name) for name in _SOURCE_INTEGRATION_TENSOR_FIELDS}
        object.__setattr__(
            self,
            "evidence_sha256",
            _source_integration_digest(source_sha, dt_bits, tensors),
        )
        object.__setattr__(self, "_sealed", True)

    def validate_immutable(self) -> None:
        """Reauthenticate the source arithmetic against its canonical bytes."""
        if self._sealed is not True:
            raise RuntimeError("source integration evidence is not sealed")
        if not _is_canonical_sha256(self.source_transition_sha256):
            raise RuntimeError("source integration transition identity changed type or value")
        if type(self.dt_seconds) is not float or type(self.dt_float32_bits) is not str:
            raise RuntimeError("source integration timestep changed type after authentication")
        if not _is_canonical_sha256(self.evidence_sha256):
            raise RuntimeError("source integration evidence identity changed type or value")
        dt32 = np.float32(self.dt_seconds)
        dt_bits = f"0x{np.asarray(dt32).view(np.uint32).item():08x}"
        if (
            not np.isfinite(dt32)
            or dt32 <= np.float32(0.0)
            or float(dt32) != float(self.dt_seconds)
            or dt_bits != self.dt_float32_bits
        ):
            raise RuntimeError("source integration timestep changed after authentication")
        tensors = {name: object.__getattribute__(self, name) for name in _SOURCE_INTEGRATION_TENSOR_FIELDS}
        if (
            _source_integration_digest(self.source_transition_sha256, self.dt_float32_bits, tensors)
            != self.evidence_sha256
        ):
            raise RuntimeError("source integration evidence changed after authentication")

    def _owned_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(object.__getattribute__(self, name) for name in _SOURCE_INTEGRATION_TENSOR_FIELDS)


def _physical_step_digest(
    tensors: Mapping[str, torch.Tensor],
    integration_policy: str,
    source_evidence_sha256: str | None,
) -> str:
    digest = hashlib.sha256(b"pr2901-v5-physical-step-context-v2\0")
    policy = json.dumps(
        {
            "integration_policy": integration_policy,
            "source_evidence_sha256": source_evidence_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(policy).to_bytes(8, "big"))
    digest.update(policy)
    for name in _PHYSICAL_STEP_TENSOR_FIELDS:
        value = tensors[name].detach().contiguous()
        _update_tensor_digest(digest, name, value)
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
        integration_policy: Registered arithmetic used to bind the learned
            history and loads to the common-objective inertial target.
        source_evidence: Exact authentication-only SolverVBD source inputs for
            the staged-float32 policy, or ``None`` for the algebraic policy.
    """

    x_current: torch.Tensor
    x_previous: torch.Tensor
    force: torch.Tensor
    gravity: torch.Tensor
    mu: torch.Tensor
    lam: torch.Tensor
    pin: torch.Tensor
    pinned_targets: torch.Tensor
    integration_policy: str = PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64
    source_evidence: SolverVBDStagedFloat32Evidence | None = None
    physical_step_sha256: str = dataclasses.field(init=False)
    _sealed: bool = dataclasses.field(init=False, repr=False, default=False)

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in _PHYSICAL_STEP_TENSOR_FIELDS and object.__getattribute__(self, "_sealed"):
            return value.clone()
        return value

    def __post_init__(self) -> None:
        if type(self.integration_policy) is not str or self.integration_policy not in _PHYSICAL_INTEGRATION_POLICIES:
            raise ValueError("integration_policy is not registered")
        if self.integration_policy == PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64:
            if self.source_evidence is not None:
                raise ValueError("algebraic integration policy must not carry SolverVBD source evidence")
        elif type(self.source_evidence) is not SolverVBDStagedFloat32Evidence:
            raise TypeError("SolverVBD integration policy requires canonical source evidence")
        else:
            self.source_evidence.validate_immutable()

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
        if self.source_evidence is not None:
            source_positions, *_ = self.source_evidence._owned_tensors()
            if self.x_current.ndim != 2:
                raise ValueError("SolverVBD source integration evidence supports only one unbatched transition")
            if position_dtype != torch.float64:
                raise ValueError("SolverVBD source integration policy requires promoted float64 runtime tensors")
            if source_positions.shape != self.x_current.shape or source_positions.device != device:
                raise ValueError("source integration evidence must match the physical position shape and device")

        for name in _PHYSICAL_STEP_TENSOR_FIELDS:
            object.__setattr__(self, name, getattr(self, name).clone())
        tensors = {name: getattr(self, name) for name in _PHYSICAL_STEP_TENSOR_FIELDS}
        evidence_sha256 = None if self.source_evidence is None else self.source_evidence.evidence_sha256
        object.__setattr__(
            self,
            "physical_step_sha256",
            _physical_step_digest(tensors, self.integration_policy, evidence_sha256),
        )
        object.__setattr__(self, "_sealed", True)

    def validate_immutable(self) -> None:
        """Reauthenticate the sealed context against its canonical bytes."""
        if self._sealed is not True:
            raise RuntimeError("physical-step context is not sealed")
        if type(self.integration_policy) is not str or self.integration_policy not in _PHYSICAL_INTEGRATION_POLICIES:
            raise RuntimeError("physical integration policy changed type or value")
        if not _is_canonical_sha256(self.physical_step_sha256):
            raise RuntimeError("physical-step identity changed type or value")
        if self.integration_policy == PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64:
            if self.source_evidence is not None:
                raise RuntimeError("algebraic physical step gained source integration evidence")
        elif type(self.source_evidence) is not SolverVBDStagedFloat32Evidence:
            raise RuntimeError("SolverVBD physical step lost its canonical source evidence")
        else:
            self.source_evidence.validate_immutable()
        tensors = {name: object.__getattribute__(self, name) for name in _PHYSICAL_STEP_TENSOR_FIELDS}
        evidence_sha256 = None if self.source_evidence is None else self.source_evidence.evidence_sha256
        if _physical_step_digest(tensors, self.integration_policy, evidence_sha256) != self.physical_step_sha256:
            raise RuntimeError("physical-step context changed after authentication")

    def _validate_sealed(self) -> None:
        """Check construction state without repeating cold byte authentication."""
        if not self._sealed:
            raise RuntimeError("physical-step context is not sealed")

    def _owned_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return zero-copy owned tensors for internal solver code."""
        return tuple(object.__getattribute__(self, name) for name in _PHYSICAL_STEP_TENSOR_FIELDS)


@dataclasses.dataclass(frozen=True)
class ProposalSafeguardConfig:
    """Versionable fixed-work proposal-globalization semantics.

    Args:
        policy: Registered top-level safeguard policy.
        candidate_step_fractions: Unique built-in floats in strictly descending
            order from exactly ``1.0`` to exactly ``0.0``.
        interpolation_policy: Registered candidate construction policy.
        selection_policy: Registered deterministic selection policy.
        zero_policy: Registered zero-candidate behavior.
        candidate_state_policy: Registered constraint-state branching policy.
            Registered candidate hooks must treat the prepared input state as
            immutable and return a separately owned successor state. Tensor
            inputs are defensively copied, but opaque Python state cannot be
            copied generically by the solver.
    """

    candidate_step_fractions: tuple[float, ...]
    policy: str = "fixed-constrained-backtracking-v1"
    interpolation_policy: str = "current-to-projected-position-segment"
    selection_policy: str = "first-admissible-positive-else-zero"
    zero_policy: str = "exact-no-op"
    candidate_state_policy: str = "same-prepared-state-selected-successor"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate exact policy types and values at the execution boundary."""
        fractions = self.candidate_step_fractions
        if type(fractions) is not tuple or len(fractions) < 2:
            raise TypeError("candidate_step_fractions must be a tuple containing at least 1.0 and 0.0")
        if any(type(fraction) is not float for fraction in fractions):
            raise TypeError("candidate_step_fractions entries must be built-in floats")
        if any(not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0 for fraction in fractions):
            raise ValueError("candidate_step_fractions entries must be finite and lie in [0, 1]")
        if fractions[0] != 1.0 or fractions[-1] != 0.0:
            raise ValueError("candidate_step_fractions must start at exactly 1.0 and end at exactly 0.0")
        if any(fractions[index] <= fractions[index + 1] for index in range(len(fractions) - 1)):
            raise ValueError("candidate_step_fractions must be unique and strictly descending")
        policies = {
            "policy": "fixed-constrained-backtracking-v1",
            "interpolation_policy": "current-to-projected-position-segment",
            "selection_policy": "first-admissible-positive-else-zero",
            "zero_policy": "exact-no-op",
            "candidate_state_policy": "same-prepared-state-selected-successor",
        }
        for name, expected in policies.items():
            if type(getattr(self, name)) is not str or getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r}")


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
        proposal_safeguard: Optional fixed inference-only proposal safeguard.
            Safeguard mode requires an unbatched state and strict
            objective/residual policies. Every configured candidate is
            constrained and scored before deterministic selection.
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
    proposal_safeguard: ProposalSafeguardConfig | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate all solver semantics at an execution boundary."""
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
        if self.proposal_safeguard is not None:
            if type(self.proposal_safeguard) is not ProposalSafeguardConfig:
                raise TypeError("proposal_safeguard must be a ProposalSafeguardConfig")
            self.proposal_safeguard.validate()
            if self.objective_policy != "require-nonincreasing" or self.residual_policy != "require-nonincreasing":
                raise ValueError("proposal safeguard requires strict objective and residual nonincrease policies")


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

    def constrain_candidate(
        self,
        state: object,
        iteration: int,
        candidate_index: int,
        candidate_step_fraction: float,
        positions: torch.Tensor,
        candidate_positions: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> ConstraintApplication:
        """Constrain one fixed globalization candidate without committing it.

        Registered implementations must treat ``state`` and all tensor inputs
        as immutable. Every candidate receives the same prepared state; only
        the separately owned state returned for the selected candidate is
        advanced to the next iteration.
        """


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

    def constrain_candidate(
        self,
        state: object,
        iteration: int,
        candidate_index: int,
        candidate_step_fraction: float,
        positions: torch.Tensor,
        candidate_positions: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> ConstraintApplication:
        """Apply identity semantics independently to one fixed candidate."""
        del candidate_index, candidate_step_fraction
        return self.constrain(state, iteration, positions, candidate_positions, pinned, pinned_targets)


@dataclasses.dataclass(frozen=True)
class CandidateEvaluation:
    """One constrained and fully scored fixed globalization candidate."""

    candidate_index: int
    step_fraction: float
    candidate_positions: torch.Tensor
    constrained_positions: torch.Tensor
    normalized_residual: torch.Tensor
    raw_residual_norm: torch.Tensor
    normalized_residual_norm: torch.Tensor
    objective: torch.Tensor
    minimum_determinant: torch.Tensor
    minimum_singular_value: torch.Tensor
    positions_finite: bool
    exact_pins: bool
    determinant_valid: bool
    singular_value_valid: bool
    objective_finite: bool
    residual_finite: bool
    state_valid: bool
    objective_nonincreasing: bool
    residual_nonincreasing: bool
    zero_step_unchanged: bool | None
    displacement_retention: torch.Tensor | None
    learned_contribution_retained: bool
    admissible: bool
    rejection_reasons: tuple[str, ...]
    constraint_diagnostics: Mapping[str, object]


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
    candidate_evaluations: tuple[CandidateEvaluation, ...] | None = None
    selected_candidate_index: int | None = None
    selected_step_fraction: float | None = None
    learned_displacement_retention: torch.Tensor | None = None
    learned_contribution_retained: bool | None = None
    proposal_accepted: bool | None = None
    selection_reason: str | None = None


@dataclasses.dataclass
class IterativeSolverResult:
    """Final constrained iterate, residual, work, and full learned trace.

    ``operator_geometry_sha256``, ``projection_state_sha256``, and
    ``static_graph_sha256`` identify the
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
    physical_integration_policy: str
    source_integration_evidence_sha256: str | None
    physical_step_sha256: str
    common_objective_sha256: str
    operator_geometry_sha256: str
    projection_state_sha256: str
    static_graph_sha256: str
    objective: torch.Tensor
    raw_residual_norm: torch.Tensor
    normalized_residual_norm: torch.Tensor
    minimum_determinant: torch.Tensor
    minimum_singular_value: torch.Tensor
    proposal_accepted_iterations: int | None = None
    zero_step_iterations: int | None = None
    learned_contribution_retained_iterations: int | None = None


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


def _tensor_bytes_equal(left: object, right: object) -> bool:
    """Compare exact tensor metadata and contiguous storage bytes."""
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if (
        left.shape != right.shape
        or left.device != right.device
        or left.dtype != right.dtype
        or left.layout != torch.strided
        or right.layout != torch.strided
    ):
        return False
    left_bytes = left.contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


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
    torch_solver.validate_authenticated_operator_geometry(projection_state)
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
    _validate_translation_gauge_objective_binding(projection_state, objective)


def _validate_translation_gauge_objective_binding(
    projection_state: SolverState,
    objective: CommonObjectiveContext,
) -> None:
    """Bind a free-body projection's translation gauge to objective mass."""
    if projection_state.translation_gauge_policy != torch_solver.TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS:
        if projection_state.center_of_mass_weights is not None:
            raise ValueError("projection translation gauge policy and center-of-mass weights disagree")
        return

    weights = projection_state.center_of_mass_weights
    if (
        projection_state.projection_backend != "dense"
        or projection_state.tikhonov != 0.0
        or projection_state.pinned.numel() != 0
        or weights is None
        or weights.layout != torch.strided
        or weights.shape != (projection_state.n_verts,)
        or weights.device != objective.device
        or weights.dtype != objective.dtype
        or not weights.is_floating_point()
    ):
        raise ValueError("mass-weighted center-of-mass projection state is not canonical")
    expected_free = torch.arange(projection_state.n_verts, dtype=torch.int64, device=objective.device)
    if not torch.equal(projection_state.free, expected_free):
        raise ValueError("mass-weighted center-of-mass projection must keep every vertex free")
    mass = objective._owned_tensor("mass")
    expected_weights = mass / mass.sum()
    if not torch.equal(weights, expected_weights):
        raise ValueError("projection center-of-mass weights differ from normalized common-objective mass")


def _exact_float32_image(name: str, value: torch.Tensor) -> np.ndarray:
    """Return an exact float32 source image or reject a lossy runtime value."""
    runtime = value.detach().contiguous().cpu().numpy()
    source = np.asarray(runtime, dtype=np.float32)
    if not _array_bytes_equal(runtime, source.astype(runtime.dtype)):
        raise ValueError(f"{name} must be an exact promotion of its float32 source value")
    return np.array(source, dtype=np.float32, order="C", copy=True)


def validate_projection_objective_volume_binding(
    projection_state: SolverState,
    objective: CommonObjectiveContext,
) -> None:
    """Require one exact portable volume identity across projection and objective."""
    if type(projection_state) is not SolverState:
        raise TypeError("projection_state must be a canonical SolverState")
    if type(objective) is not CommonObjectiveContext:
        raise TypeError("objective must be a canonical CommonObjectiveContext")
    portable_projection = (
        projection_state.operator_geometry_policy
        == torch_solver.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME
    )
    objective_binding = (
        objective.operator_geometry_sha256,
        objective.operator_volume_policy,
        objective.operator_volume_sha256,
    )
    bound_objective = all(value is not None for value in objective_binding)
    if portable_projection:
        if not bound_objective:
            raise ValueError("portable projection requires a bound common-objective volume identity")
        expected = (
            projection_state.operator_geometry_sha256,
            projection_state.operator_volume_policy,
            projection_state.operator_volume_sha256,
        )
        if objective_binding != expected:
            raise ValueError("projection and common-objective portable volume identities differ")
    elif any(value is not None for value in objective_binding):
        raise ValueError("non-portable projection cannot be paired with a portable objective volume identity")


def validate_physical_objective_integration(
    projection_state: SolverState,
    objective: CommonObjectiveContext,
    physical_step: PhysicalStepContext,
) -> None:
    """Authenticate learned history/load inputs against one common objective.

    The generic policy retains the execution-dtype algebraic identity used by
    synthetic fixtures. The SolverVBD policy instead replays the independently
    rounded source history and forward step in canonical CPU float32. The two
    float32 expressions are deliberately checked separately; one cannot be
    inverted to prove the other after rounding.
    """
    physical_step.validate_immutable()
    objective.validate_immutable()
    torch_solver.validate_authenticated_operator_geometry(projection_state)
    if projection_state.projection_state_sha256 != torch_solver.projection_state_sha256(projection_state):
        raise ValueError("compatibility projection state differs from its authenticated identity")
    if projection_state.n_verts != objective.n_vertices or projection_state.n_tets != objective.n_tets:
        raise ValueError("projection and common-objective mesh sizes differ")
    if projection_state.rest_q.device != objective.device or projection_state.rest_q.dtype != objective.dtype:
        raise ValueError("projection and common objective must share device and dtype")
    for name, projected, common in (
        ("tets", projection_state.tets, objective._owned_tensor("tets")),
        ("J", projection_state.J, objective._owned_tensor("J")),
        ("volume", projection_state.w, objective._owned_tensor("volume")),
        ("pinned", projection_state.pinned, objective._owned_tensor("pinned")),
    ):
        if not torch.equal(projected, common):
            raise ValueError(f"projection and common-objective {name} differ")
    validate_projection_objective_volume_binding(projection_state, objective)
    _validate_translation_gauge_objective_binding(projection_state, objective)
    _validate_physical_objective_integration_trusted(projection_state, objective, physical_step)


def _validate_physical_objective_integration_trusted(
    projection_state: SolverState,
    objective: CommonObjectiveContext,
    physical_step: PhysicalStepContext,
) -> None:
    """Validate integration semantics after caller-owned identity checks."""
    physical_step._validate_sealed()
    x_current, x_previous, force, gravity, mu, lam, pin, pinned_targets = physical_step._owned_tensors()
    if x_current.ndim not in (2, 3) or x_current.shape[-2:] != (projection_state.n_verts, 3):
        raise ValueError("physical positions must end in the exact projection vertex shape")
    if x_previous.shape != x_current.shape or force.shape != x_current.shape:
        raise ValueError("physical history and force must exactly match x_current")
    if any(
        value.device != objective.device or value.dtype != objective.dtype for value in (x_current, x_previous, force)
    ):
        raise ValueError("physical positions, history, and force must share the common-objective device and dtype")
    expected_gravity_shapes = ((3,), (*x_current.shape[:-2], 3))
    if (
        gravity.shape not in expected_gravity_shapes
        or gravity.device != objective.device
        or gravity.dtype != objective.dtype
    ):
        raise ValueError("physical gravity has the wrong shape, device, or dtype")
    expected_material_shapes = ((projection_state.n_tets,),)
    if x_current.ndim == 3:
        expected_material_shapes += ((x_current.shape[0], projection_state.n_tets),)
    if mu.shape not in expected_material_shapes or lam.shape != mu.shape or pin.shape != mu.shape:
        raise ValueError("physical material and pin features have the wrong exact tet shape")
    if any(value.device != objective.device or value.dtype != objective.dtype for value in (mu, lam, pin)):
        raise ValueError("physical material and pin features must share the common-objective device and dtype")
    targets = _expand_pinned_targets(x_current, projection_state.pinned, pinned_targets)
    if not torch.equal(x_current[..., projection_state.pinned, :], targets):
        raise ValueError("physical current state does not contain the exact pinned targets")
    for name, model_value in (("mu", mu), ("lam", lam)):
        objective_value = objective._owned_tensor(name)
        expected = objective_value.expand_as(model_value)
        if not torch.equal(model_value, expected):
            raise ValueError(f"model {name} features differ from the bound common objective")

    expected_pin = torch.isin(projection_state.tets, projection_state.pinned).any(dim=-1).to(pin)
    expected_pin = expected_pin.expand_as(pin)
    if not torch.equal(pin, expected_pin):
        raise ValueError("model pin-incidence features differ from the projection constraints")

    if physical_step.integration_policy == PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32:
        evidence = physical_step.source_evidence
        if type(evidence) is not SolverVBDStagedFloat32Evidence:
            raise TypeError("SolverVBD physical integration policy lost its source evidence")
        evidence.validate_immutable()
        if objective.dtype != torch.float64 or x_current.dtype != torch.float64:
            raise ValueError("SolverVBD source integration policy requires promoted float64 runtime tensors")
        if x_current.ndim != 2:
            raise ValueError("SolverVBD physical integration policy requires one unbatched transition")
        pre_event, velocity, source_mass, inverse_mass = evidence._owned_tensors()
        if any(value.device != objective.device for value in (pre_event, velocity, source_mass, inverse_mass)):
            raise ValueError("SolverVBD source evidence must share the common-objective device")
        if objective.dt != evidence.dt_seconds:
            raise ValueError("SolverVBD source timestep differs from the common objective")

        mass = objective._owned_tensor("mass")
        expected_mass = source_mass.to(dtype=mass.dtype)
        mass_runtime = mass.detach().contiguous().cpu().numpy()
        mass_expected = expected_mass.detach().contiguous().cpu().numpy()
        if not _array_bytes_equal(mass_runtime, mass_expected):
            raise ValueError("common-objective mass differs from the exact SolverVBD source mass")

        pinned = projection_state.pinned.detach().cpu().numpy().astype(np.int64, copy=False)
        pre_event32 = pre_event.detach().contiguous().cpu().numpy().copy()
        velocity32 = velocity.detach().contiguous().cpu().numpy().copy()
        inverse_mass32 = inverse_mass.detach().contiguous().cpu().numpy().copy()
        x_current32 = _exact_float32_image("x_current", x_current)
        x_previous32 = _exact_float32_image("x_previous", x_previous)
        force32 = _exact_float32_image("force", force)
        gravity32 = _exact_float32_image("gravity", gravity)
        pin_targets32 = _exact_float32_image("pinned_targets", pinned_targets)
        if pin_targets32.shape != (pinned.size, 3):
            raise ValueError("SolverVBD pinned_targets must have shape (P, 3)")

        expected_current = pre_event32.copy()
        expected_current[pinned] = pin_targets32
        if not _array_bytes_equal(x_current32, expected_current):
            raise ValueError("x_current differs from the exact SolverVBD applied callback state")

        dt32 = np.float32(evidence.dt_seconds)
        displacement = (velocity32 * dt32).astype(np.float32)
        expected_previous = (pre_event32 - displacement).astype(np.float32)
        if not _array_bytes_equal(x_previous32, expected_previous):
            raise ValueError("x_previous differs from the exact SolverVBD float32 history construction")

        force_acceleration = (force32 * inverse_mass32[:, None]).astype(np.float32)
        acceleration = (gravity32[None, :] + force_acceleration).astype(np.float32)
        velocity_new = (velocity32 + (acceleration * dt32).astype(np.float32)).astype(np.float32)
        expected_target = (expected_current + (velocity_new * dt32).astype(np.float32)).astype(np.float32)
        expected_target[pinned] = expected_current[pinned]
        inertial_target = objective._owned_tensor("inertial_target")
        inertial_target32 = _exact_float32_image("common-objective inertial_target", inertial_target)
        if not _array_bytes_equal(inertial_target32, expected_target):
            raise ValueError("common-objective inertial target differs from the staged SolverVBD float32 replay")
        return

    if physical_step.integration_policy != PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64:
        raise ValueError("physical integration policy is not registered")
    if objective.dtype != torch.float64 or x_current.dtype != torch.float64:
        raise ValueError("algebraic-float64 integration policy requires float64 runtime tensors")
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


# Kept as the internal spelling used by the solver and existing ablation
# harness after they have already authenticated the complete problem.
_validate_physical_context = _validate_physical_objective_integration_trusted


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
    if config.proposal_safeguard is not None:
        if type(config.proposal_safeguard) is not ProposalSafeguardConfig:
            raise TypeError("proposal_safeguard must remain a ProposalSafeguardConfig at execution")
        config.proposal_safeguard.validate()
        fractions = reference.new_tensor(config.proposal_safeguard.candidate_step_fractions)
        if not bool(torch.isfinite(fractions).all().item()):
            raise ValueError(f"candidate_step_fractions must remain finite in execution dtype {reference.dtype}")
        if not bool((fractions[0] == reference.new_tensor(1.0)).item()) or not bool(
            (fractions[-1] == reference.new_tensor(0.0)).item()
        ):
            raise ValueError(f"candidate_step_fractions endpoints changed in execution dtype {reference.dtype}")
        if not bool((fractions[:-1] > fractions[1:]).all().item()):
            raise ValueError(
                f"candidate_step_fractions must remain unique and strictly descending in execution dtype {reference.dtype}"
            )


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


def _candidate_geometry_metrics(
    projection_state: SolverState,
    positions: torch.Tensor,
    pinned_targets: torch.Tensor,
    minimum_determinant_bound: float,
    minimum_singular_value_bound: float,
) -> tuple[torch.Tensor, torch.Tensor, bool, bool, bool, bool, bool]:
    """Measure candidate geometry without aborting the fixed schedule."""
    nan = positions.new_full((), torch.nan)
    positions_finite = bool(torch.isfinite(positions).all().item())
    targets = _expand_pinned_targets(positions, projection_state.pinned, pinned_targets)
    exact_pins = torch.equal(positions[..., projection_state.pinned, :], targets)
    minimum_determinant = nan
    minimum_singular_value = nan
    determinant_valid = False
    singular_value_valid = False
    if positions_finite:
        deformation_gradient = torch_solver.compute_F(positions, projection_state.tets, projection_state.J)
        determinant = torch.linalg.det(deformation_gradient)
        determinant_finite = bool(torch.isfinite(determinant).all().item())
        if determinant_finite:
            minimum_determinant = determinant.amin()
            determinant_valid = bool((determinant > minimum_determinant_bound).all().item())
        if bool(torch.isfinite(deformation_gradient).all().item()):
            try:
                singular_values = torch.linalg.svdvals(deformation_gradient)
            except RuntimeError:
                singular_values = None
            if singular_values is not None and bool(torch.isfinite(singular_values).all().item()):
                minimum_singular_value = singular_values.amin()
                singular_value_valid = bool((singular_values > minimum_singular_value_bound).all().item())
    state_valid = positions_finite and exact_pins and determinant_valid and singular_value_valid
    return (
        minimum_determinant,
        minimum_singular_value,
        positions_finite,
        exact_pins,
        determinant_valid,
        singular_value_valid,
        state_valid,
    )


def _score_candidate(
    *,
    projection_state: SolverState,
    objective: CommonObjectiveContext,
    config: IterativeSolverConfig,
    current: torch.Tensor,
    objective_before: torch.Tensor,
    normalized_residual_norm_before: torch.Tensor,
    pinned_targets: torch.Tensor,
    candidate_index: int,
    step_fraction: float,
    projected_positions: torch.Tensor,
    candidate_positions: torch.Tensor,
    application: ConstraintApplication,
) -> CandidateEvaluation:
    """Score one constrained candidate under the registered strict gates."""
    constrained = application.positions
    (
        minimum_determinant,
        minimum_singular_value,
        positions_finite,
        exact_pins,
        determinant_valid,
        singular_value_valid,
        state_valid,
    ) = _candidate_geometry_metrics(
        projection_state,
        constrained,
        pinned_targets,
        config.minimum_determinant,
        config.minimum_singular_value,
    )
    raw_residual = _common_objective_residual_trusted(
        objective,
        constrained,
        detach=config.detach_residual_features,
    )
    normalized_residual = raw_residual / objective.residual_scale
    raw_residual_norm = _normalized_residual_norm(raw_residual)
    normalized_residual_norm = _normalized_residual_norm(normalized_residual)
    objective_value = _common_objective_components_trusted(objective, constrained)["total"]
    objective_finite = bool(torch.isfinite(objective_value).all().item())
    residual_finite = (
        bool(torch.isfinite(normalized_residual).all().item())
        and bool(torch.isfinite(raw_residual_norm).all().item())
        and bool(torch.isfinite(normalized_residual_norm).all().item())
    )
    objective_nonincreasing = objective_finite and bool(
        (objective_value <= objective_before + objective_value.new_tensor(config.objective_increase_tolerance))
        .all()
        .item()
    )
    residual_nonincreasing = residual_finite and bool(
        (
            normalized_residual_norm
            <= normalized_residual_norm_before
            + normalized_residual_norm.new_tensor(config.normalized_residual_increase_tolerance)
        )
        .all()
        .item()
    )
    zero_step_unchanged = _tensor_bytes_equal(constrained, current) if step_fraction == 0.0 else None
    free = torch.ones(projection_state.n_verts, dtype=torch.bool, device=current.device)
    free[projection_state.pinned] = False
    full_displacement = (projected_positions - current)[free].reshape(-1)
    constrained_displacement = (constrained - current)[free].reshape(-1)
    full_displacement_finite = bool(torch.isfinite(full_displacement).all().item())
    constrained_displacement_finite = bool(torch.isfinite(constrained_displacement).all().item())
    if (
        not full_displacement_finite
        or not constrained_displacement_finite
        or torch.equal(full_displacement, torch.zeros_like(full_displacement))
    ):
        displacement_retention = None
    else:
        retention_numerator = torch.dot(constrained_displacement, full_displacement)
        retention_denominator = torch.dot(full_displacement, full_displacement)
        if (
            not bool(torch.isfinite(retention_numerator).item())
            or not bool(torch.isfinite(retention_denominator).item())
            or not bool((retention_denominator > 0.0).item())
        ):
            displacement_retention = None
        else:
            candidate_retention = retention_numerator / retention_denominator
            displacement_retention = candidate_retention if bool(torch.isfinite(candidate_retention).item()) else None
    learned_contribution_retained = (
        step_fraction > 0.0
        and displacement_retention is not None
        and bool(torch.isfinite(displacement_retention).item())
        and bool((displacement_retention > 0.0).item())
    )
    rejection_reasons: list[str] = []
    if not positions_finite:
        rejection_reasons.append("non-finite-positions")
    if not exact_pins:
        rejection_reasons.append("changed-exact-pins")
    if not determinant_valid:
        rejection_reasons.append("determinant-bound")
    if not singular_value_valid:
        rejection_reasons.append("singular-value-bound")
    if not objective_finite:
        rejection_reasons.append("non-finite-objective")
    if not residual_finite:
        rejection_reasons.append("non-finite-residual")
    if not objective_nonincreasing:
        rejection_reasons.append("objective-increase")
    if not residual_nonincreasing:
        rejection_reasons.append("residual-increase")
    if zero_step_unchanged is False:
        rejection_reasons.append("zero-step-moved")
    admissible = state_valid and objective_nonincreasing and residual_nonincreasing and zero_step_unchanged is not False
    return CandidateEvaluation(
        candidate_index=candidate_index,
        step_fraction=step_fraction,
        candidate_positions=candidate_positions,
        constrained_positions=constrained,
        normalized_residual=normalized_residual,
        raw_residual_norm=raw_residual_norm,
        normalized_residual_norm=normalized_residual_norm,
        objective=objective_value,
        minimum_determinant=minimum_determinant,
        minimum_singular_value=minimum_singular_value,
        positions_finite=positions_finite,
        exact_pins=exact_pins,
        determinant_valid=determinant_valid,
        singular_value_valid=singular_value_valid,
        objective_finite=objective_finite,
        residual_finite=residual_finite,
        state_valid=state_valid,
        objective_nonincreasing=objective_nonincreasing,
        residual_nonincreasing=residual_nonincreasing,
        zero_step_unchanged=zero_step_unchanged,
        displacement_retention=displacement_retention,
        learned_contribution_retained=learned_contribution_retained,
        admissible=admissible,
        rejection_reasons=tuple(rejection_reasons),
        constraint_diagnostics=dict(application.diagnostics),
    )


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
        Final state, normalized residual, per-iteration trace, exact work
        counts, and the constraint descriptor. The direct path remains fully
        differentiable. Proposal-safeguard selection is an inference/research
        path with deterministic host-visible gate decisions.
    """
    if type(config) is not IterativeSolverConfig:
        raise TypeError("config must be the exact IterativeSolverConfig type")
    config.validate()
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
    safeguard = config.proposal_safeguard
    if safeguard is not None and x_initial.ndim != 2:
        raise ValueError("proposal safeguard v1 requires an unbatched position state")
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
    constrain_candidate = getattr(hook, "constrain_candidate", None)
    if safeguard is not None and not callable(constrain_candidate):
        raise TypeError("proposal safeguard requires a candidate-aware constrain_candidate hook")
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
    if safeguard is None:
        constraint_state = hook.begin_step(current, projection_state.pinned, pinned_targets)
    else:
        constraint_state = hook.begin_step(current.clone(), projection_state.pinned.clone(), pinned_targets.clone())
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
    _validate_finite("initial raw residual norm", raw_residual_norm)
    _validate_finite("initial normalized residual norm", normalized_residual_norm)
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
        iteration_positions = current if safeguard is None else current.clone()
        observation_positions = iteration_positions if safeguard is None else iteration_positions.clone()
        observation = hook.prepare_iteration(constraint_state, iteration, observation_positions)
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
            iteration_positions,
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
            projection_kwargs["initial_positions"] = iteration_positions
        if projection_state.translation_gauge_policy == torch_solver.TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS:
            projection_kwargs["center_of_mass_positions"] = objective._owned_tensor("inertial_target")
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
        if safeguard is None:
            _validate_finite("projected proposal", proposed)
        elif (
            proposed.shape != iteration_positions.shape
            or proposed.device != iteration_positions.device
            or proposed.dtype != iteration_positions.dtype
        ):
            raise RuntimeError("projected proposal changed the position shape, device, or dtype")
        trace_proposed = proposed if safeguard is None else proposed.clone()

        candidate_evaluations: tuple[CandidateEvaluation, ...] | None = None
        selected_candidate_index: int | None = None
        selected_step_fraction: float | None = None
        learned_displacement_retention: torch.Tensor | None = None
        learned_contribution_retained: bool | None = None
        proposal_accepted: bool | None = None
        selection_reason: str | None = None
        if safeguard is None:
            application = hook.constrain(
                constraint_state,
                iteration,
                iteration_positions,
                proposed,
                projection_state.pinned,
                pinned_targets,
            )
            _reauthenticate_contexts(physical_step, objective)
            if not isinstance(application, ConstraintApplication):
                raise RuntimeError("constraint hook constrain must return ConstraintApplication")
            constraint_state = application.state
            committed = application.positions
            if (
                committed.shape != iteration_positions.shape
                or committed.device != iteration_positions.device
                or committed.dtype != iteration_positions.dtype
            ):
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
            _validate_finite("committed raw residual norm", raw_residual_norm_after)
            _validate_finite("committed normalized residual norm", normalized_residual_norm_after)
            objective_after = _common_objective_components_trusted(objective, committed)["total"]
            _validate_finite("committed common objective", objective_after)
        else:
            applications: list[ConstraintApplication] = []
            evaluations: list[CandidateEvaluation] = []
            projected_displacement = trace_proposed - iteration_positions
            for candidate_index, step_fraction in enumerate(safeguard.candidate_step_fractions):
                if step_fraction == 1.0:
                    candidate_positions = trace_proposed.clone()
                elif step_fraction == 0.0:
                    candidate_positions = iteration_positions.clone()
                else:
                    candidate_positions = (
                        iteration_positions + iteration_positions.new_tensor(step_fraction) * projected_displacement
                    )
                hook_positions = iteration_positions.clone()
                hook_candidate_positions = candidate_positions.clone()
                application = constrain_candidate(
                    constraint_state,
                    iteration,
                    candidate_index,
                    step_fraction,
                    hook_positions,
                    hook_candidate_positions,
                    projection_state.pinned.clone(),
                    pinned_targets.clone(),
                )
                _reauthenticate_contexts(physical_step, objective)
                if not isinstance(application, ConstraintApplication):
                    raise RuntimeError("constraint hook constrain_candidate must return ConstraintApplication")
                constrained = application.positions
                if (
                    constrained.shape != iteration_positions.shape
                    or constrained.device != iteration_positions.device
                    or constrained.dtype != iteration_positions.dtype
                ):
                    raise RuntimeError("constraint candidate changed the position shape, device, or dtype")
                application = dataclasses.replace(application, positions=constrained.clone())
                applications.append(application)
                evaluations.append(
                    _score_candidate(
                        projection_state=projection_state,
                        objective=objective,
                        config=config,
                        current=iteration_positions,
                        objective_before=objective_value,
                        normalized_residual_norm_before=normalized_residual_norm,
                        pinned_targets=pinned_targets,
                        candidate_index=candidate_index,
                        step_fraction=step_fraction,
                        projected_positions=trace_proposed,
                        candidate_positions=candidate_positions,
                        application=application,
                    )
                )
            candidate_evaluations = tuple(evaluations)
            zero_evaluation = candidate_evaluations[-1]
            if zero_evaluation.zero_step_unchanged is not True:
                raise RuntimeError("zero-step constraint candidate must return the current iterate bitwise")
            selected_candidate_index = next(
                (item.candidate_index for item in candidate_evaluations[:-1] if item.admissible),
                None,
            )
            if selected_candidate_index is None:
                if not zero_evaluation.admissible:
                    reasons = ", ".join(zero_evaluation.rejection_reasons)
                    raise RuntimeError(f"zero-step fallback is not admissible: {reasons}")
                selected_candidate_index = zero_evaluation.candidate_index
                selection_reason = "no-admissible-positive"
            selected_evaluation = candidate_evaluations[selected_candidate_index]
            selected_application = applications[selected_candidate_index]
            selected_step_fraction = selected_evaluation.step_fraction
            proposal_accepted = selected_step_fraction > 0.0
            learned_displacement_retention = selected_evaluation.displacement_retention
            learned_contribution_retained = selected_evaluation.learned_contribution_retained
            if selection_reason is None:
                if learned_displacement_retention is None:
                    selection_reason = "first-admissible-positive-candidate-zero-projected-displacement"
                elif learned_contribution_retained:
                    selection_reason = "first-admissible-positive-candidate"
                else:
                    selection_reason = "first-admissible-positive-candidate-no-learned-displacement"
            constraint_state = selected_application.state
            committed = selected_evaluation.constrained_positions.clone()
            residual_after = selected_evaluation.normalized_residual
            raw_residual_norm_after = selected_evaluation.raw_residual_norm
            normalized_residual_norm_after = selected_evaluation.normalized_residual_norm
            objective_after = selected_evaluation.objective
            minimum_determinant_after = selected_evaluation.minimum_determinant
            minimum_singular_value_after = selected_evaluation.minimum_singular_value
            application = selected_application
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
                positions_before=iteration_positions,
                normalized_residual_before=residual,
                raw_residual_norm_before=raw_residual_norm,
                normalized_residual_norm_before=normalized_residual_norm,
                objective_before=objective_value,
                minimum_determinant_before=minimum_determinant,
                minimum_singular_value_before=minimum_singular_value,
                delta_h=delta_h,
                omega=omega,
                target_deformation_gradient=target_f,
                proposed_positions=trace_proposed,
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
                candidate_evaluations=candidate_evaluations,
                selected_candidate_index=selected_candidate_index,
                selected_step_fraction=selected_step_fraction,
                learned_displacement_retention=learned_displacement_retention,
                learned_contribution_retained=learned_contribution_retained,
                proposal_accepted=proposal_accepted,
                selection_reason=selection_reason,
            )
        )
        current = committed if safeguard is None else committed.clone()
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

    candidate_applications_per_iteration = 1 if safeguard is None else len(safeguard.candidate_step_fractions)
    if safeguard is None:
        proposal_accepted_iterations = None
        zero_step_iterations = None
        learned_contribution_retained_iterations = None
    else:
        proposal_accepted_iterations = sum(item.proposal_accepted is True for item in trace)
        zero_step_iterations = sum(item.selected_step_fraction == 0.0 for item in trace)
        learned_contribution_retained_iterations = sum(item.learned_contribution_retained is True for item in trace)

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
            residual_evaluations=config.iterations * candidate_applications_per_iteration + 1,
            objective_evaluations=config.iterations * candidate_applications_per_iteration + 1,
            state_validity_evaluations=config.iterations * candidate_applications_per_iteration + 1,
            constraint_preparations=config.iterations,
            constraint_applications=config.iterations * candidate_applications_per_iteration,
            physical_step_authentications=config.iterations * (candidate_applications_per_iteration + 1) + 3,
            common_objective_authentications=config.iterations * (candidate_applications_per_iteration + 1) + 3,
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
        physical_integration_policy=physical_step.integration_policy,
        source_integration_evidence_sha256=(
            None if physical_step.source_evidence is None else physical_step.source_evidence.evidence_sha256
        ),
        physical_step_sha256=physical_step.physical_step_sha256,
        common_objective_sha256=objective.common_objective_sha256,
        operator_geometry_sha256=projection_state.operator_geometry_sha256,
        projection_state_sha256=projection_state.projection_state_sha256,
        static_graph_sha256=predictor.model.static_graph_sha256,
        objective=objective_value,
        raw_residual_norm=raw_residual_norm,
        normalized_residual_norm=normalized_residual_norm,
        minimum_determinant=minimum_determinant,
        minimum_singular_value=minimum_singular_value,
        proposal_accepted_iterations=proposal_accepted_iterations,
        zero_step_iterations=zero_step_iterations,
        learned_contribution_retained_iterations=learned_contribution_retained_iterations,
    )


__all__ = [
    "PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64",
    "PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32",
    "CandidateEvaluation",
    "ConstraintApplication",
    "ConstraintObservation",
    "IdentityConstraintHook",
    "IterationConstraintHook",
    "IterativeSolverConfig",
    "IterativeSolverIteration",
    "IterativeSolverResult",
    "IterativeSolverWork",
    "PhysicalStepContext",
    "ProposalSafeguardConfig",
    "SolverVBDStagedFloat32Evidence",
    "solve_iterative_principal_stretch",
    "validate_physical_objective_integration",
    "validate_projection_objective_volume_binding",
]
