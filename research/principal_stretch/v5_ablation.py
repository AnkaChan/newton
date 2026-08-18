# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Development-only identical-corrector ablations for iterative v5.

This harness compares independently produced starting points under one shared
fixed-work physics corrector.  The learned, zero-head, and permuted-head rows
all execute the architecture-v5 predictor and compatibility projection for
the same ``K`` iterations from the same authenticated physical state.  The
persistence and force-shifted inertial rows are derived internally.  A VBD-K1
row is admitted only through a sealed, exact-one-sweep caller attestation.

The VBD attestation binds bytes and identities, but Python cannot prove that
an external VBD implementation actually ran.  Consequently every report is
explicitly development-only and unregistered: it is neither checkpoint replay
evidence, DAT evidence, promotion evidence, nor a learned-value claim.  A
learned-value claim requires held-out results showing a benefit over the same
correction pipeline; this module only makes that comparison auditable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import time

import torch

from . import torch_solver
from .iterative_solver import (
    CandidateEvaluation,
    IterativeSolverConfig,
    IterativeSolverIteration,
    IterativeSolverResult,
    IterativeSolverWork,
    PhysicalStepContext,
    ProposalSafeguardConfig,
    _expand_pinned_targets,
    _validate_physical_context,
    _validate_problem_identity,
    solve_iterative_principal_stretch,
)
from .predictor import StretchPredictor, predictor_architecture_version
from .torch_solver import SolverState
from .v5_checkpoint import (
    _verify_predictor_execution_surface,
    canonical_json_sha256,
    learned_state_sha256,
)
from .v5_corrector import (
    CorrectorCandidateTrace,
    CorrectorConfig,
    CorrectorTrace,
    CorrectorWork,
    FixedPCGConfig,
    correct_common_objective,
)
from .v5_objective import CommonObjectiveContext, common_objective_components, common_objective_residual

LEARNED_ARM = "v5-learned"
ZERO_ARM = "v5-zero-head"
PERMUTED_ARM = "v5-permuted-head"
PERSISTENCE_ARM = "persistence"
INERTIAL_ARM = "force-shifted-inertial"
VBD_K1_ARM = "caller-attested-vbd-k1"

MANDATORY_ARM_NAMES = (
    LEARNED_ARM,
    ZERO_ARM,
    PERMUTED_ARM,
    PERSISTENCE_ARM,
    INERTIAL_ARM,
    VBD_K1_ARM,
)

_CLAIM_SCOPE = "development-only-unregistered-identical-corrector-ablation-no-learned-value-claim"
_VBD_FRESHNESS_SCOPE = "caller-attested-vbd-k1-not-execution-verified"
_CHECKPOINT_SCOPE = "not-bound-to-v5-checkpoint-runtime-replay"
_DAT_SCOPE = "collision-free-identity-constraint-only-no-dat-claim"
_TIMING_SCOPE = "host-wall-dispatch-not-device-synchronized-not-performance-evidence"
_VBD_METHOD = "newton-vbd"
_VBD_INITIALIZER = "authenticated-physical-step-x-current"
_DEFAULT_PROPOSAL_OBJECTIVE_INCREASE_TOLERANCE = 1.0e-12
_DEFAULT_PROPOSAL_NORMALIZED_RESIDUAL_INCREASE_TOLERANCE = 1.0e-12
_MAX_PROPOSAL_INCREASE_TOLERANCE = 1.0e-6
_ARM_ORIGINS = {
    LEARNED_ARM: "fresh-v5-learned-k-from-authenticated-physical-persistence",
    ZERO_ARM: "fresh-v5-zero-head-k-with-model-and-projection-executed",
    PERMUTED_ARM: "fresh-v5-nonidentity-permuted-head-k-with-model-and-projection-executed",
    PERSISTENCE_ARM: "authenticated-physical-x-current-validated-exact-pins-no-overwrite",
    INERTIAL_ARM: "common-objective-force-shifted-inertial-target-with-explicit-exact-pin-overwrite",
    VBD_K1_ARM: "external-vbd-k1-caller-attested-not-execution-verified",
}


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _update_tensor_digest(digest: object, name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must have strided layout")
    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    canonical = value.detach().contiguous()
    metadata = json.dumps(
        {"name": name, "dtype": str(canonical.dtype), "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = canonical.view(torch.uint8).cpu().numpy().tobytes()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def position_sha256(positions: torch.Tensor) -> str:
    """Hash one finite unbatched position field including dtype and shape."""
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be a torch.Tensor with shape (V, 3)")
    if not positions.is_floating_point():
        raise ValueError("positions must have a floating dtype")
    digest = hashlib.sha256(b"pr2901-v5-ablation-position-v1\0")
    _update_tensor_digest(digest, "positions", positions)
    return digest.hexdigest()


def pin_binding_sha256(pinned: torch.Tensor, pinned_targets: torch.Tensor) -> str:
    """Hash the ordered Dirichlet indices and exact physical targets."""
    if not isinstance(pinned, torch.Tensor) or pinned.ndim != 1 or pinned.dtype != torch.int64:
        raise ValueError("pinned must be an int64 tensor with shape (P,)")
    if pinned.numel() > 0 and torch.unique(pinned).numel() != pinned.numel():
        raise ValueError("pinned must not contain duplicates")
    if (
        not isinstance(pinned_targets, torch.Tensor)
        or pinned_targets.shape != (pinned.numel(), 3)
        or not pinned_targets.is_floating_point()
    ):
        raise ValueError("pinned_targets must be a floating tensor with shape (P, 3)")
    digest = hashlib.sha256(b"pr2901-v5-ablation-pin-binding-v1\0")
    _update_tensor_digest(digest, "pinned", pinned)
    _update_tensor_digest(digest, "pinned_targets", pinned_targets)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class VBDK1MethodRecord:
    """Exact caller statement for an external fresh one-sweep VBD endpoint.

    ``source_run_sha256`` identifies the external execution receipt.  The
    fixed fields prevent a generic named tensor from masquerading as VBD-K1,
    but this record still cannot authenticate that arbitrary Python executed
    the declared method.
    """

    source_run_sha256: str
    schema_version: int = 1
    method: str = _VBD_METHOD
    implicit_step_count: int = 1
    solver_sweeps: int = 1
    initial_state_source: str = _VBD_INITIALIZER
    warm_start: bool = False
    history_advance_count: int = 0
    method_record_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.source_run_sha256, "source_run_sha256")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("VBD-K1 method record requires schema_version=1")
        if self.method != _VBD_METHOD:
            raise ValueError(f"VBD-K1 method must be {_VBD_METHOD!r}")
        if (
            type(self.implicit_step_count) is not int
            or type(self.solver_sweeps) is not int
            or self.implicit_step_count != 1
            or self.solver_sweeps != 1
        ):
            raise ValueError("VBD-K1 method record requires exactly one implicit step and one solver sweep")
        if self.initial_state_source != _VBD_INITIALIZER:
            raise ValueError("VBD-K1 must initialize from the authenticated physical x_current")
        if not isinstance(self.warm_start, bool) or self.warm_start:
            raise ValueError("VBD-K1 method record forbids warm starts")
        if type(self.history_advance_count) is not int or self.history_advance_count != 0:
            raise ValueError("the ablation VBD-K1 row must not advance shared history")
        object.__setattr__(self, "method_record_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "implicit_step_count": self.implicit_step_count,
            "solver_sweeps": self.solver_sweeps,
            "initial_state_source": self.initial_state_source,
            "warm_start": self.warm_start,
            "history_advance_count": self.history_advance_count,
            "source_run_sha256": self.source_run_sha256,
            "freshness_scope": _VBD_FRESHNESS_SCOPE,
        }

    def validate_immutable(self) -> None:
        """Recompute the exact method-record identity."""
        if canonical_json_sha256(self._payload()) != self.method_record_sha256:
            raise RuntimeError("VBD-K1 method record changed after authentication")


def _vbd_attestation_payload(value: AttestedVBDK1Start) -> dict[str, object]:
    return {
        "schema_version": 2,
        "positions_sha256": object.__getattribute__(value, "positions_sha256"),
        "physical_step_sha256": object.__getattribute__(value, "physical_step_sha256"),
        "common_objective_sha256": object.__getattribute__(value, "common_objective_sha256"),
        "static_mesh_sha256": object.__getattribute__(value, "static_mesh_sha256"),
        "operator_geometry_sha256": object.__getattribute__(value, "operator_geometry_sha256"),
        "projection_state_sha256": object.__getattribute__(value, "projection_state_sha256"),
        "pin_binding_sha256": object.__getattribute__(value, "pin_binding_sha256"),
        "method_record_sha256": object.__getattribute__(value, "method_record").method_record_sha256,
        "freshness_scope": _VBD_FRESHNESS_SCOPE,
    }


@dataclasses.dataclass(frozen=True)
class AttestedVBDK1Start:
    """Owned VBD-K1 endpoint bound to the exact ablation problem identities."""

    positions: torch.Tensor
    physical_step_sha256: str
    common_objective_sha256: str
    static_mesh_sha256: str
    operator_geometry_sha256: str
    projection_state_sha256: str
    pin_binding_sha256: str
    method_record: VBDK1MethodRecord
    positions_sha256: str = dataclasses.field(init=False)
    attestation_sha256: str = dataclasses.field(init=False)
    _sealed: bool = dataclasses.field(init=False, repr=False, default=False)

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name == "positions" and object.__getattribute__(self, "_sealed"):
            return value.clone()
        return value

    def __post_init__(self) -> None:
        if not isinstance(self.positions, torch.Tensor) or self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("VBD-K1 positions must have shape (V, 3)")
        if not self.positions.is_floating_point() or not torch.isfinite(self.positions).all():
            raise ValueError("VBD-K1 positions must have a finite floating dtype")
        for name in (
            "physical_step_sha256",
            "common_objective_sha256",
            "static_mesh_sha256",
            "operator_geometry_sha256",
            "projection_state_sha256",
            "pin_binding_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.method_record) is not VBDK1MethodRecord:
            raise TypeError("method_record must be a VBDK1MethodRecord")
        self.method_record.validate_immutable()
        owned = self.positions.detach().clone()
        object.__setattr__(self, "positions", owned)
        object.__setattr__(self, "positions_sha256", position_sha256(owned))
        object.__setattr__(self, "attestation_sha256", canonical_json_sha256(_vbd_attestation_payload(self)))
        object.__setattr__(self, "_sealed", True)

    def _owned_positions(self) -> torch.Tensor:
        return object.__getattribute__(self, "positions")

    def validate_immutable(self) -> None:
        """Reauthenticate the sealed endpoint and exact method statement."""
        if not self._sealed:
            raise RuntimeError("VBD-K1 attestation is not sealed")
        self.method_record.validate_immutable()
        if position_sha256(self._owned_positions()) != self.positions_sha256:
            raise RuntimeError("VBD-K1 positions changed after authentication")
        if canonical_json_sha256(_vbd_attestation_payload(self)) != self.attestation_sha256:
            raise RuntimeError("VBD-K1 attestation changed after authentication")


@dataclasses.dataclass(frozen=True)
class V5AblationConfig:
    """Fixed learned work and authenticated nonidentity head permutation."""

    iterations: int
    head_permutation: tuple[int, ...]
    detach_residual_features: bool = True
    minimum_determinant: float = 0.0
    minimum_singular_value: float = 0.0
    proposal_safeguard: ProposalSafeguardConfig | None = None
    proposal_objective_increase_tolerance: float = _DEFAULT_PROPOSAL_OBJECTIVE_INCREASE_TOLERANCE
    proposal_normalized_residual_increase_tolerance: float = _DEFAULT_PROPOSAL_NORMALIZED_RESIDUAL_INCREASE_TOLERANCE

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, numbers.Integral):
            raise TypeError("iterations must be an integer")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not isinstance(self.head_permutation, tuple) or not self.head_permutation:
            raise ValueError("head_permutation must be a non-empty tuple")
        if any(isinstance(index, bool) or not isinstance(index, numbers.Integral) for index in self.head_permutation):
            raise TypeError("head_permutation entries must be integers")
        if not isinstance(self.detach_residual_features, bool):
            raise TypeError("detach_residual_features must be a bool")
        for name in ("minimum_determinant", "minimum_singular_value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.proposal_safeguard is not None:
            if type(self.proposal_safeguard) is not ProposalSafeguardConfig:
                raise TypeError("proposal_safeguard must be a ProposalSafeguardConfig")
            self.proposal_safeguard.validate()
        for name in (
            "proposal_objective_increase_tolerance",
            "proposal_normalized_residual_increase_tolerance",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a built-in float")
            if not math.isfinite(value) or value < 0.0 or value > _MAX_PROPOSAL_INCREASE_TOLERANCE:
                raise ValueError(f"{name} must be a registered finite non-negative tolerance")
        if self.proposal_safeguard is None and (
            self.proposal_objective_increase_tolerance != _DEFAULT_PROPOSAL_OBJECTIVE_INCREASE_TOLERANCE
            or self.proposal_normalized_residual_increase_tolerance
            != _DEFAULT_PROPOSAL_NORMALIZED_RESIDUAL_INCREASE_TOLERANCE
        ):
            raise ValueError("proposal tolerances may change only when proposal_safeguard is enabled")


@dataclasses.dataclass(frozen=True)
class EndpointMetrics:
    """Independently evaluated common-objective and geometry metrics."""

    positions_sha256: str
    total_objective: float
    inertia_objective: float
    elastic_objective: float
    raw_residual_norm: float
    normalized_residual_norm: float
    minimum_determinant: float
    maximum_determinant: float
    minimum_singular_value: float
    inverted_tet_count: int
    maximum_pin_error: float
    exact_pins: bool


@dataclasses.dataclass(frozen=True)
class AblationTiming:
    """Host-dispatch timing explicitly excluded from performance claims."""

    pre_corrector_seconds: float
    start_scoring_seconds: float
    corrector_seconds: float
    endpoint_scoring_seconds: float
    scope: str = _TIMING_SCOPE


@dataclasses.dataclass(frozen=True)
class AblationArmResult:
    """One independently initialized row under the shared corrector."""

    name: str
    start_origin: str
    evidence_scope: str
    vbd_freshness_scope: str | None
    head_mode: str | None
    head_permutation: tuple[int, ...] | None
    physical_step_sha256: str
    physical_integration_policy: str
    source_integration_evidence_sha256: str | None
    common_objective_sha256: str
    static_mesh_sha256: str
    operator_geometry_sha256: str
    static_graph_sha256: str
    projection_state_sha256: str
    pin_binding_sha256: str
    predictor_state_sha256: str
    corrector_config_sha256: str
    corrector_scheduled_work_sha256: str
    vbd_attestation_sha256: str | None
    vbd_method_record_sha256: str | None
    pre_corrector_positions: torch.Tensor
    corrected_positions: torch.Tensor
    pre_corrector_metrics: EndpointMetrics
    corrected_metrics: EndpointMetrics
    iterative_work: IterativeSolverWork | None
    attested_vbd_sweeps: int
    corrector_trace: CorrectorTrace
    corrector_calls: int
    pin_overwrite_vertices: int
    correction_norm: float
    correction_rms: float
    start_displacement_from_persistence_norm: float
    corrected_displacement_from_persistence_norm: float
    initializer_displacement_retention: float | None
    learned_displacement_retention: float | None
    fallback_preserved_start: bool
    timing: AblationTiming
    proposal_safeguard_config_sha256: str | None = None
    proposal_trace: tuple[IterativeSolverIteration, ...] | None = None
    solver_proposal_displacement_retention: tuple[float | None, ...] | None = None
    proposal_accepted_iterations: int | None = None
    zero_step_iterations: int | None = None
    learned_contribution_retained_iterations: int | None = None


@dataclasses.dataclass(frozen=True)
class V5AblationResult:
    """Six mandatory rows and their deliberately limited claim scope."""

    schema_version: int
    claim_scope: str
    development_only: bool
    learned_value_claim: bool
    checkpoint_scope: str
    dat_scope: str
    vbd_freshness_scope: str
    physical_step_sha256: str
    physical_integration_policy: str
    source_integration_evidence_sha256: str | None
    common_objective_sha256: str
    static_mesh_sha256: str
    operator_geometry_sha256: str
    projection_state_sha256: str
    static_graph_sha256: str
    predictor_state_sha256: str
    pin_binding_sha256: str
    ablation_config: V5AblationConfig
    ablation_config_sha256: str
    corrector_config: CorrectorConfig
    corrector_config_sha256: str
    corrector_scheduled_work_sha256: str
    learned_scheduled_work_sha256: str
    head_permutation_sha256: str
    iterations: int
    pinned_vertex_count: int
    corrector_call_count: int
    arms: tuple[AblationArmResult, ...]
    evidence_sha256: str
    proposal_safeguard_config_sha256: str | None = None
    proposal_pinned_indices: torch.Tensor | None = None
    proposal_pinned_targets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate_immutable()

    def validate_immutable(self) -> None:
        """Revalidate mutable endpoint bytes and the complete evidence digest."""
        names = tuple(arm.name for arm in self.arms)
        if names != MANDATORY_ARM_NAMES or len(set(names)) != len(names):
            raise ValueError("ablation result must contain each mandatory canonical row exactly once in order")
        if type(self.ablation_config) is not V5AblationConfig:
            raise TypeError("ablation result must embed the exact V5AblationConfig")
        self.ablation_config.__post_init__()
        candidate_mode = self.ablation_config.proposal_safeguard is not None
        expected_schema_version = 4 if candidate_mode else 3
        if (
            type(self.schema_version) is not int
            or self.schema_version != expected_schema_version
            or self.claim_scope != _CLAIM_SCOPE
        ):
            raise ValueError("ablation result has an unsupported claim schema")
        if not self.development_only or self.learned_value_claim:
            raise ValueError("this harness may report only development diagnostics without a learned-value claim")
        if self.corrector_call_count != len(MANDATORY_ARM_NAMES):
            raise ValueError("every mandatory row must receive exactly one corrector call")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations <= 0
            or isinstance(self.pinned_vertex_count, bool)
            or not isinstance(self.pinned_vertex_count, int)
            or self.pinned_vertex_count < 0
        ):
            raise ValueError("ablation work and pinned counts must be canonical integers")
        if self.checkpoint_scope != _CHECKPOINT_SCOPE or self.dat_scope != _DAT_SCOPE:
            raise ValueError("ablation result changed its checkpoint or DAT limitation")
        if self.vbd_freshness_scope != _VBD_FRESHNESS_SCOPE:
            raise ValueError("ablation result changed its VBD freshness limitation")
        if type(self.physical_integration_policy) is not str or self.physical_integration_policy not in (
            "algebraic-float64-position-history-loads-v1",
            "solver-vbd-staged-float32-v1",
        ):
            raise ValueError("ablation result changed to an unregistered physical integration policy")
        if self.physical_integration_policy == "algebraic-float64-position-history-loads-v1":
            if self.source_integration_evidence_sha256 is not None:
                raise ValueError("algebraic ablation result must not name source integration evidence")
        else:
            if type(self.source_integration_evidence_sha256) is not str:
                raise TypeError("source_integration_evidence_sha256 must be canonical text")
            _require_sha256(self.source_integration_evidence_sha256, "source_integration_evidence_sha256")
        for name in (
            "physical_step_sha256",
            "common_objective_sha256",
            "static_mesh_sha256",
            "operator_geometry_sha256",
            "projection_state_sha256",
            "static_graph_sha256",
            "predictor_state_sha256",
            "pin_binding_sha256",
            "ablation_config_sha256",
            "corrector_config_sha256",
            "corrector_scheduled_work_sha256",
            "learned_scheduled_work_sha256",
            "head_permutation_sha256",
            "evidence_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.corrector_config) is not CorrectorConfig or type(self.corrector_config.pcg) is not FixedPCGConfig:
            raise TypeError("ablation result must embed the exact CorrectorConfig")
        if _ablation_config_sha256(self.ablation_config) != self.ablation_config_sha256:
            raise RuntimeError("embedded ablation configuration changed after authentication")
        if _config_sha256(self.corrector_config) != self.corrector_config_sha256:
            raise RuntimeError("embedded corrector configuration changed after authentication")
        if self.iterations != self.ablation_config.iterations:
            raise RuntimeError("ablation iteration count differs from its embedded configuration")
        if any(arm.corrector_calls != 1 for arm in self.arms):
            raise ValueError("every ablation row must record exactly one corrector call")
        if any(arm.corrector_config_sha256 != self.corrector_config_sha256 for arm in self.arms):
            raise ValueError("all rows must bind the identical corrector configuration")
        if any(arm.corrector_scheduled_work_sha256 != self.corrector_scheduled_work_sha256 for arm in self.arms):
            raise ValueError("all rows must execute identical scheduled corrector work")
        vbd_row = self.arms[-1]
        if vbd_row.vbd_freshness_scope != _VBD_FRESHNESS_SCOPE:
            raise ValueError("the VBD row must expose its caller-attested freshness limitation")

        expected_heads = {
            LEARNED_ARM: ("learned", None),
            ZERO_ARM: ("zero", None),
            PERMUTED_ARM: ("permuted", self.arm(PERMUTED_ARM).head_permutation),
            PERSISTENCE_ARM: (None, None),
            INERTIAL_ARM: (None, None),
            VBD_K1_ARM: (None, None),
        }
        permutation = self.arm(PERMUTED_ARM).head_permutation
        if (
            permutation is None
            or permutation != self.ablation_config.head_permutation
            or _permutation_sha256(permutation) != self.head_permutation_sha256
        ):
            raise RuntimeError("permuted row differs from the authenticated head permutation")
        persistence = self.arm(PERSISTENCE_ARM).pre_corrector_positions
        if candidate_mode:
            assert self.ablation_config.proposal_safeguard is not None
            _validate_candidate_execution_scalars(self.ablation_config, persistence)
            expected_safeguard_sha256 = _proposal_safeguard_config_sha256(self.ablation_config.proposal_safeguard)
            if (
                type(self.proposal_safeguard_config_sha256) is not str
                or self.proposal_safeguard_config_sha256 != expected_safeguard_sha256
            ):
                raise RuntimeError("candidate safeguard configuration changed after authentication")
            if (
                type(self.proposal_pinned_indices) is not torch.Tensor
                or self.proposal_pinned_indices.dtype != torch.int64
                or self.proposal_pinned_indices.ndim != 1
                or self.proposal_pinned_indices.numel() != self.pinned_vertex_count
                or type(self.proposal_pinned_targets) is not torch.Tensor
                or self.proposal_pinned_targets.shape != (self.pinned_vertex_count, 3)
                or self.proposal_pinned_targets.device != persistence.device
                or self.proposal_pinned_targets.dtype != persistence.dtype
                or self.proposal_pinned_indices.device != persistence.device
            ):
                raise RuntimeError("candidate pin evidence changed shape, dtype, or device")
            if self.proposal_pinned_indices.numel() > 0 and (
                bool((self.proposal_pinned_indices < 0).any().item())
                or bool((self.proposal_pinned_indices >= persistence.shape[0]).any().item())
            ):
                raise RuntimeError("candidate pin evidence contains an out-of-range vertex")
            if pin_binding_sha256(
                self.proposal_pinned_indices, self.proposal_pinned_targets
            ) != self.pin_binding_sha256 or not torch.equal(
                persistence.index_select(0, self.proposal_pinned_indices),
                self.proposal_pinned_targets,
            ):
                raise RuntimeError("candidate pin evidence changed after authentication")
        elif any(
            value is not None
            for value in (
                self.proposal_safeguard_config_sha256,
                self.proposal_pinned_indices,
                self.proposal_pinned_targets,
            )
        ):
            raise RuntimeError("legacy direct ablation acquired candidate safeguard evidence")
        for arm in self.arms:
            if type(arm.physical_integration_policy) is not str:
                raise RuntimeError("ablation row changed its physical integration policy type")
            if (
                arm.source_integration_evidence_sha256 is not None
                and type(arm.source_integration_evidence_sha256) is not str
            ):
                raise RuntimeError("ablation row changed its source integration evidence type")
            expected_mode, expected_permutation = expected_heads[arm.name]
            if arm.head_mode != expected_mode or arm.head_permutation != expected_permutation:
                raise RuntimeError("ablation row changed its canonical head semantics")
            if arm.evidence_scope != self.claim_scope:
                raise RuntimeError("ablation row changed its development-only evidence scope")
            if arm.start_origin != _ARM_ORIGINS[arm.name]:
                raise RuntimeError("ablation row changed its canonical start origin")
            for name in (
                "physical_step_sha256",
                "physical_integration_policy",
                "source_integration_evidence_sha256",
                "common_objective_sha256",
                "static_mesh_sha256",
                "operator_geometry_sha256",
                "static_graph_sha256",
                "projection_state_sha256",
                "pin_binding_sha256",
                "predictor_state_sha256",
                "corrector_config_sha256",
                "corrector_scheduled_work_sha256",
            ):
                expected = getattr(self, name)
                if getattr(arm, name) != expected:
                    raise RuntimeError(f"ablation row changed its {name}")
            if position_sha256(arm.pre_corrector_positions) != arm.pre_corrector_metrics.positions_sha256:
                raise RuntimeError("ablation pre-corrector positions changed after hashing")
            if position_sha256(arm.corrected_positions) != arm.corrected_metrics.positions_sha256:
                raise RuntimeError("ablation corrected positions changed after hashing")
            for metrics in (arm.pre_corrector_metrics, arm.corrected_metrics):
                _require_sha256(metrics.positions_sha256, "endpoint positions_sha256")
                numeric = (
                    metrics.total_objective,
                    metrics.inertia_objective,
                    metrics.elastic_objective,
                    metrics.raw_residual_norm,
                    metrics.normalized_residual_norm,
                    metrics.minimum_determinant,
                    metrics.maximum_determinant,
                    metrics.minimum_singular_value,
                    metrics.maximum_pin_error,
                )
                if not all(math.isfinite(value) for value in numeric):
                    raise RuntimeError("ablation endpoint metrics changed to non-finite data")
                if not metrics.exact_pins or metrics.maximum_pin_error != 0.0:
                    raise RuntimeError("ablation endpoint metrics no longer certify exact pins")
                if (
                    isinstance(metrics.inverted_tet_count, bool)
                    or not isinstance(metrics.inverted_tet_count, int)
                    or metrics.inverted_tet_count < 0
                ):
                    raise RuntimeError("ablation inversion count is not canonical")
            if arm.corrector_trace.common_objective_sha256 != self.common_objective_sha256:
                raise RuntimeError("ablation corrector trace changed its objective identity")
            if not _metric_matches(
                arm.corrector_trace.start_objective,
                arm.pre_corrector_metrics.total_objective,
                arm.pre_corrector_positions.dtype,
            ) or not _metric_matches(
                arm.corrector_trace.start_raw_residual_norm,
                arm.pre_corrector_metrics.raw_residual_norm,
                arm.pre_corrector_positions.dtype,
            ):
                raise RuntimeError("ablation corrector trace changed its independently scored start")
            if (
                not arm.corrector_trace.candidates
                or arm.corrector_trace.work.candidate_count != len(arm.corrector_trace.candidates)
                or not 0 <= arm.corrector_trace.selected_candidate_index < len(arm.corrector_trace.candidates)
            ):
                raise RuntimeError("ablation corrector trace changed its candidate schedule")
            selected = arm.corrector_trace.candidates[arm.corrector_trace.selected_candidate_index]
            if (
                selected.index != arm.corrector_trace.selected_candidate_index
                or selected.alpha != arm.corrector_trace.selected_alpha
            ):
                raise RuntimeError("ablation corrector trace changed its selected candidate")
            if (
                tuple(candidate.alpha for candidate in arm.corrector_trace.candidates)
                != self.corrector_config.candidate_alphas
            ):
                raise RuntimeError("ablation corrector trace changed its configured candidate alphas")
            for candidate in arm.corrector_trace.candidates:
                (
                    determinant_valid,
                    singular_value_valid,
                    objective_valid,
                    residual_valid,
                    expected_admissible,
                ) = _replay_candidate_safeguards(
                    candidate,
                    arm.corrector_trace,
                    self.corrector_config,
                    arm.pre_corrector_positions,
                )
                if (
                    candidate.determinant_valid != determinant_valid
                    or candidate.singular_value_valid != singular_value_valid
                    or candidate.objective_nonincreasing != objective_valid
                    or candidate.residual_nonincreasing != residual_valid
                    or candidate.admissible_nonzero_step != expected_admissible
                ):
                    raise RuntimeError("ablation corrector candidate changed its configured safeguard semantics")
            if _corrector_scheduled_work_sha256(arm.corrector_trace.work) != self.corrector_scheduled_work_sha256:
                raise RuntimeError("ablation corrector trace changed its scheduled work")
            if arm.corrector_trace.accepted:
                if (
                    arm.fallback_preserved_start
                    or arm.corrector_trace.reason != "accepted"
                    or selected.alpha <= 0.0
                    or not selected.admissible_nonzero_step
                    or not _metric_matches(
                        selected.objective,
                        arm.corrected_metrics.total_objective,
                        arm.corrected_positions.dtype,
                    )
                    or not _metric_matches(
                        selected.raw_residual_norm,
                        arm.corrected_metrics.raw_residual_norm,
                        arm.corrected_positions.dtype,
                    )
                    or not _metric_matches(
                        selected.minimum_determinant,
                        arm.corrected_metrics.minimum_determinant,
                        arm.corrected_positions.dtype,
                    )
                    or not _metric_matches(
                        selected.minimum_singular_value,
                        arm.corrected_metrics.minimum_singular_value,
                        arm.corrected_positions.dtype,
                    )
                ):
                    raise RuntimeError("accepted ablation row has inconsistent fallback metadata")
            elif (
                selected.alpha != 0.0
                or not arm.fallback_preserved_start
                or arm.pre_corrector_metrics.positions_sha256 != arm.corrected_metrics.positions_sha256
                or not torch.equal(arm.pre_corrector_positions, arm.corrected_positions)
            ):
                raise RuntimeError("rejected ablation row no longer preserves its exact start")
            correction = arm.corrected_positions - arm.pre_corrector_positions
            correction_norm = float(torch.linalg.vector_norm(correction).detach().cpu())
            correction_rms = float(torch.sqrt(correction.square().mean()).detach().cpu())
            if not _metric_matches(correction_norm, arm.correction_norm, arm.corrected_positions.dtype):
                raise RuntimeError("ablation correction norm changed after authentication")
            if not _metric_matches(correction_rms, arm.correction_rms, arm.corrected_positions.dtype):
                raise RuntimeError("ablation correction RMS changed after authentication")
            start_norm, corrected_norm, retention = _retention(
                arm.pre_corrector_positions,
                arm.corrected_positions,
                persistence,
            )
            if not _metric_matches(
                start_norm,
                arm.start_displacement_from_persistence_norm,
                arm.corrected_positions.dtype,
            ) or not _metric_matches(
                corrected_norm,
                arm.corrected_displacement_from_persistence_norm,
                arm.corrected_positions.dtype,
            ):
                raise RuntimeError("ablation persistence displacement norm changed after authentication")
            if retention is None:
                if arm.initializer_displacement_retention is not None:
                    raise RuntimeError("ablation initializer retention changed after authentication")
            elif arm.initializer_displacement_retention is None or not _metric_matches(
                retention,
                arm.initializer_displacement_retention,
                arm.corrected_positions.dtype,
            ):
                raise RuntimeError("ablation initializer retention changed after authentication")
            expected_learned_retention = retention if arm.name == LEARNED_ARM else None
            if arm.learned_displacement_retention != expected_learned_retention:
                raise RuntimeError("ablation learned-displacement retention changed after authentication")
            timing_values = (
                arm.timing.pre_corrector_seconds,
                arm.timing.start_scoring_seconds,
                arm.timing.corrector_seconds,
                arm.timing.endpoint_scoring_seconds,
            )
            if arm.timing.scope != _TIMING_SCOPE or not all(
                math.isfinite(value) and value >= 0.0 for value in timing_values
            ):
                raise RuntimeError("ablation timing changed outside its diagnostic scope")

            is_iterative = arm.name in (LEARNED_ARM, ZERO_ARM, PERMUTED_ARM)
            if is_iterative:
                if arm.iterative_work is None:
                    raise RuntimeError("learned ablation row lost its predictor/projection work")
                if (
                    canonical_json_sha256(_iterative_scheduled_work_payload(arm.iterative_work))
                    != self.learned_scheduled_work_sha256
                ):
                    raise RuntimeError("learned ablation row changed its scheduled predictor/projection work")
                if (
                    arm.iterative_work.predictor_passes != self.iterations
                    or arm.iterative_work.projection_calls != self.iterations
                ):
                    raise RuntimeError("learned ablation row changed its fixed K work")
                if candidate_mode:
                    if (
                        type(arm.proposal_safeguard_config_sha256) is not str
                        or arm.proposal_safeguard_config_sha256 != self.proposal_safeguard_config_sha256
                    ):
                        raise RuntimeError("candidate row changed its shared safeguard configuration")
                    assert self.ablation_config.proposal_safeguard is not None
                    _validate_candidate_trace(
                        arm,
                        safeguard=self.ablation_config.proposal_safeguard,
                        minimum_determinant=float(self.ablation_config.minimum_determinant),
                        minimum_singular_value=float(self.ablation_config.minimum_singular_value),
                        objective_increase_tolerance=self.ablation_config.proposal_objective_increase_tolerance,
                        normalized_residual_increase_tolerance=(
                            self.ablation_config.proposal_normalized_residual_increase_tolerance
                        ),
                        iterations=self.iterations,
                        persistence=persistence,
                        pinned=self.proposal_pinned_indices,
                        pinned_targets=self.proposal_pinned_targets,
                    )
                elif any(
                    value is not None
                    for value in (
                        arm.proposal_safeguard_config_sha256,
                        arm.proposal_trace,
                        arm.solver_proposal_displacement_retention,
                        arm.proposal_accepted_iterations,
                        arm.zero_step_iterations,
                        arm.learned_contribution_retained_iterations,
                    )
                ):
                    raise RuntimeError("legacy direct ablation row acquired unbound candidate evidence")
            elif arm.iterative_work is not None:
                raise RuntimeError("classical initializer row acquired learned work")
            elif any(
                value is not None
                for value in (
                    arm.proposal_safeguard_config_sha256,
                    arm.proposal_trace,
                    arm.solver_proposal_displacement_retention,
                    arm.proposal_accepted_iterations,
                    arm.zero_step_iterations,
                    arm.learned_contribution_retained_iterations,
                )
            ):
                raise RuntimeError("classical initializer row acquired candidate evidence")
            if arm.name == INERTIAL_ARM:
                if arm.pin_overwrite_vertices != self.pinned_vertex_count:
                    raise RuntimeError("inertial row changed its explicit pin-overwrite count")
            elif arm.pin_overwrite_vertices != 0:
                raise RuntimeError("an ablation row silently acquired pin preprocessing")
            if arm.name == VBD_K1_ARM:
                if (
                    arm.attested_vbd_sweeps != 1
                    or arm.vbd_attestation_sha256 is None
                    or arm.vbd_method_record_sha256 is None
                    or arm.vbd_freshness_scope != _VBD_FRESHNESS_SCOPE
                ):
                    raise RuntimeError("VBD row changed its caller-attested one-sweep evidence")
                _require_sha256(arm.vbd_attestation_sha256, "vbd_attestation_sha256")
                _require_sha256(arm.vbd_method_record_sha256, "vbd_method_record_sha256")
            elif (
                arm.attested_vbd_sweeps != 0
                or arm.vbd_attestation_sha256 is not None
                or arm.vbd_method_record_sha256 is not None
                or arm.vbd_freshness_scope is not None
            ):
                raise RuntimeError("non-VBD row acquired VBD evidence")

        identities = _ablation_result_identities(self)
        if _ablation_evidence_sha256(identities=identities, arms=self.arms) != self.evidence_sha256:
            raise RuntimeError("ablation evidence changed after authentication")

    def arm(self, name: str) -> AblationArmResult:
        """Return one canonical row by exact name."""
        matches = tuple(arm for arm in self.arms if arm.name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


@dataclasses.dataclass(frozen=True)
class AblationComparison:
    """Numeric left-minus-right deltas without a quality verdict."""

    left_arm: str
    right_arm: str
    start_total_objective_delta: float
    corrected_total_objective_delta: float
    start_raw_residual_norm_delta: float
    corrected_raw_residual_norm_delta: float
    corrected_minimum_determinant_delta: float
    corrected_minimum_singular_value_delta: float
    correction_norm_delta: float
    left_corrector_accepted: bool
    right_corrector_accepted: bool
    quality_verdict: None
    claim_scope: str = "development-numeric-deltas-only-no-quality-verdict"


def _score_endpoint(
    objective: CommonObjectiveContext,
    projection_state: SolverState,
    pinned_targets: torch.Tensor,
    positions: torch.Tensor,
) -> EndpointMetrics:
    if positions.shape != (objective.n_vertices, 3):
        raise ValueError(f"endpoint must have shape ({objective.n_vertices}, 3)")
    if positions.device != objective.device or positions.dtype != objective.dtype:
        raise ValueError("endpoint must share the common objective device and dtype")
    if not torch.isfinite(positions).all():
        raise RuntimeError("endpoint positions contain a non-finite value")
    components = common_objective_components(objective, positions)
    residual = common_objective_residual(objective, positions)
    raw_residual_norm_tensor = torch.linalg.vector_norm(residual)
    normalized_residual_norm_tensor = raw_residual_norm_tensor / objective.residual_scale
    deformation_gradient = torch_solver.compute_F(positions, projection_state.tets, projection_state.J)
    determinant = torch.linalg.det(deformation_gradient)
    singular_values = torch.linalg.svdvals(deformation_gradient)
    if (
        not torch.isfinite(raw_residual_norm_tensor)
        or not torch.isfinite(normalized_residual_norm_tensor)
        or not torch.isfinite(determinant).all()
        or not torch.isfinite(singular_values).all()
    ):
        raise RuntimeError("endpoint scoring produced a non-finite value")
    pinned = projection_state.pinned
    if pinned.numel() == 0:
        maximum_pin_error_tensor = positions.new_zeros(())
        exact_pins = True
    else:
        pinned_positions = positions.index_select(0, pinned)
        maximum_pin_error_tensor = (pinned_positions - pinned_targets).abs().amax()
        exact_pins = bool(torch.equal(pinned_positions, pinned_targets))
    if not exact_pins:
        raise RuntimeError("ablation endpoint changed an exact pinned target")
    return EndpointMetrics(
        positions_sha256=position_sha256(positions),
        total_objective=float(components["total"].detach().cpu()),
        inertia_objective=float(components["inertia"].detach().cpu()),
        elastic_objective=float(components["elastic"].detach().cpu()),
        raw_residual_norm=float(raw_residual_norm_tensor.detach().cpu()),
        normalized_residual_norm=float(normalized_residual_norm_tensor.detach().cpu()),
        minimum_determinant=float(determinant.amin().detach().cpu()),
        maximum_determinant=float(determinant.amax().detach().cpu()),
        minimum_singular_value=float(singular_values.amin().detach().cpu()),
        inverted_tet_count=int((determinant <= 0.0).sum().detach().cpu()),
        maximum_pin_error=float(maximum_pin_error_tensor.detach().cpu()),
        exact_pins=exact_pins,
    )


def _config_sha256(config: object) -> str:
    if not dataclasses.is_dataclass(config):
        raise TypeError("authenticated configuration must be a dataclass")
    return canonical_json_sha256(dataclasses.asdict(config))


def _ablation_config_payload(config: V5AblationConfig) -> dict[str, object]:
    """Return the versioned config payload without changing legacy bytes."""
    payload: dict[str, object] = {
        "iterations": config.iterations,
        "head_permutation": config.head_permutation,
        "detach_residual_features": config.detach_residual_features,
        "minimum_determinant": config.minimum_determinant,
        "minimum_singular_value": config.minimum_singular_value,
    }
    if config.proposal_safeguard is not None:
        payload["proposal_safeguard"] = dataclasses.asdict(config.proposal_safeguard)
        payload["proposal_objective_increase_tolerance"] = config.proposal_objective_increase_tolerance
        payload["proposal_normalized_residual_increase_tolerance"] = (
            config.proposal_normalized_residual_increase_tolerance
        )
    return payload


def _ablation_config_sha256(config: V5AblationConfig) -> str:
    return canonical_json_sha256(_ablation_config_payload(config))


def _proposal_safeguard_config_sha256(config: ProposalSafeguardConfig) -> str:
    config.validate()
    return canonical_json_sha256(dataclasses.asdict(config))


def _permutation_sha256(permutation: tuple[int, ...]) -> str:
    return canonical_json_sha256(
        {"contract": "v5-ablation-nonidentity-tet-head-permutation-v1", "permutation": permutation}
    )


def _corrector_scheduled_work_payload(work: CorrectorWork) -> dict[str, int]:
    payload = dataclasses.asdict(work)
    payload.pop("active_pcg_iterations")
    return payload


def _iterative_scheduled_work_payload(work: IterativeSolverWork) -> dict[str, object]:
    """Exclude data-dependent inner projection counts from fixed outer work."""
    payload = dataclasses.asdict(work)
    for name in (
        "projection_iterations",
        "projection_matrix_vector_products",
        "projection_preconditioner_applications",
        "projection_factor_solves",
    ):
        payload.pop(name)
    return payload


def _corrector_scheduled_work_sha256(work: CorrectorWork) -> str:
    return canonical_json_sha256(_corrector_scheduled_work_payload(work))


def _metric_matches(left: float, right: float, dtype: torch.dtype) -> bool:
    epsilon = torch.finfo(dtype).eps
    if left == right:
        return True
    scale = max(abs(left), abs(right), torch.finfo(dtype).tiny)
    return abs(left - right) <= 128.0 * epsilon * scale


def _replay_candidate_safeguards(
    candidate: CorrectorCandidateTrace,
    trace: CorrectorTrace,
    config: CorrectorConfig,
    reference: torch.Tensor,
) -> tuple[bool, bool, bool, bool, bool]:
    """Replay line-search predicates in the original execution dtype."""
    minimum_determinant = reference.new_tensor(config.minimum_determinant)
    minimum_singular_value = reference.new_tensor(config.minimum_singular_value)
    objective_tolerance = reference.new_tensor(config.objective_increase_tolerance)
    residual_tolerance = reference.new_tensor(config.residual_increase_tolerance)
    determinant_valid = bool((reference.new_tensor(candidate.minimum_determinant) > minimum_determinant).detach().cpu())
    singular_value_valid = bool(
        (reference.new_tensor(candidate.minimum_singular_value) > minimum_singular_value).detach().cpu()
    )
    objective_valid = bool(
        (reference.new_tensor(candidate.objective) <= reference.new_tensor(trace.start_objective) + objective_tolerance)
        .detach()
        .cpu()
    )
    residual_valid = bool(
        (
            reference.new_tensor(candidate.raw_residual_norm)
            <= reference.new_tensor(trace.start_raw_residual_norm) + residual_tolerance
        )
        .detach()
        .cpu()
    )
    if not config.require_residual_nonincrease:
        residual_valid = True
    admissible = (
        candidate.alpha != 0.0
        and candidate.finite
        and candidate.exact_pins
        and determinant_valid
        and singular_value_valid
        and objective_valid
        and residual_valid
        and trace.descent_direction
        and not trace.pcg.breakdown
    )
    return determinant_valid, singular_value_valid, objective_valid, residual_valid, admissible


def _validate_corrector_result(
    *,
    objective: CommonObjectiveContext,
    projection_state: SolverState,
    pinned_targets: torch.Tensor,
    start: torch.Tensor,
    start_metrics: EndpointMetrics,
    config: CorrectorConfig,
    result_positions: torch.Tensor,
    direction: torch.Tensor,
    trace: CorrectorTrace,
) -> None:
    if trace.common_objective_sha256 != objective.common_objective_sha256:
        raise RuntimeError("corrector trace is bound to a different common objective")
    if len(trace.candidates) != len(config.candidate_alphas):
        raise RuntimeError("corrector did not evaluate every fixed line-search candidate")
    if trace.work.candidate_count != len(config.candidate_alphas):
        raise RuntimeError("corrector work does not match the fixed candidate schedule")
    if not _metric_matches(trace.start_objective, start_metrics.total_objective, objective.dtype):
        raise RuntimeError("corrector start objective differs from independent scoring")
    if not _metric_matches(trace.start_raw_residual_norm, start_metrics.raw_residual_norm, objective.dtype):
        raise RuntimeError("corrector start residual differs from independent scoring")
    if direction.shape != start.shape or direction.device != start.device or direction.dtype != start.dtype:
        raise RuntimeError("corrector direction changed shape, device, or dtype")
    if not torch.isfinite(direction).all():
        raise RuntimeError("corrector direction contains a non-finite value")
    if (
        result_positions.shape != start.shape
        or result_positions.device != start.device
        or result_positions.dtype != start.dtype
    ):
        raise RuntimeError("corrector endpoint changed shape, device, or dtype")
    if trace.accepted:
        if trace.reason != "accepted" or trace.selected_alpha <= 0.0:
            raise RuntimeError("accepted corrector trace has inconsistent selection metadata")
        expected = start + start.new_tensor(trace.selected_alpha) * direction
        if projection_state.pinned.numel() > 0:
            expected = expected.index_copy(0, projection_state.pinned, pinned_targets)
        if not torch.equal(result_positions, expected):
            raise RuntimeError("accepted corrector endpoint is not its selected candidate")
        if torch.equal(result_positions, start):
            raise RuntimeError("accepted corrector endpoint silently copied its start")
    else:
        if trace.selected_alpha != 0.0 or not torch.equal(result_positions, start):
            raise RuntimeError("rejected corrector call did not preserve its exact start")


def _validate_iterative_result(
    result: IterativeSolverResult,
    *,
    head_mode: str,
    head_permutation: tuple[int, ...] | None,
    iterations: int,
    physical_step_sha256: str,
    physical_integration_policy: str,
    source_integration_evidence_sha256: str | None,
    objective: CommonObjectiveContext,
    projection_state: SolverState,
    static_graph_sha256: str,
    persistence: torch.Tensor,
) -> None:
    if len(result.trace) != iterations or result.work.predictor_passes != iterations:
        raise RuntimeError("v5 ablation arm did not execute the fixed predictor schedule")
    if result.work.projection_calls != iterations:
        raise RuntimeError("v5 ablation arm did not execute the fixed projection schedule")
    if result.head_mode != head_mode or result.head_permutation != head_permutation:
        raise RuntimeError("v5 ablation arm returned a relabelled head mode")
    if (
        type(result.physical_integration_policy) is not str
        or result.physical_integration_policy != physical_integration_policy
    ):
        raise RuntimeError("v5 ablation arm changed the physical integration policy")
    if (
        result.source_integration_evidence_sha256 is not None
        and type(result.source_integration_evidence_sha256) is not str
    ) or result.source_integration_evidence_sha256 != source_integration_evidence_sha256:
        raise RuntimeError("v5 ablation arm changed the source integration evidence")
    if result.physical_step_sha256 != physical_step_sha256:
        raise RuntimeError("v5 ablation arm changed the physical-step identity")
    if result.common_objective_sha256 != objective.common_objective_sha256:
        raise RuntimeError("v5 ablation arm changed the common-objective identity")
    if result.operator_geometry_sha256 != projection_state.operator_geometry_sha256:
        raise RuntimeError("v5 ablation arm changed the operator-geometry identity")
    if result.projection_state_sha256 != projection_state.projection_state_sha256:
        raise RuntimeError("v5 ablation arm changed the projection identity")
    if result.static_graph_sha256 != static_graph_sha256:
        raise RuntimeError("v5 ablation arm changed the static-graph identity")
    if result.constraint_registration != "registered-identity-development":
        raise RuntimeError("v5 ablation arms require the built-in identity constraint")
    if not result.trace or not torch.equal(result.trace[0].positions_before, persistence):
        raise RuntimeError("v5 ablation arm did not freshly initialize from physical persistence")


def _clone_trace_value(value: object) -> object:
    """Own every tensor in solver proposal evidence without live aliases."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{field.name: _clone_trace_value(getattr(value, field.name)) for field in dataclasses.fields(value)},
        )
    if isinstance(value, dict):
        return {key: _clone_trace_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_trace_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_trace_value(item) for item in value]
    return value


def _clone_proposal_trace(
    trace: tuple[IterativeSolverIteration, ...],
) -> tuple[IterativeSolverIteration, ...]:
    cloned = _clone_trace_value(trace)
    if type(cloned) is not tuple or any(type(item) is not IterativeSolverIteration for item in cloned):
        raise RuntimeError("failed to create an owned iterative proposal trace")
    return cloned


def _proposal_retention_value(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    if type(value) is not torch.Tensor or value.ndim != 0 or not value.is_floating_point():
        raise RuntimeError("candidate proposal retention must be a floating scalar tensor or None")
    if not torch.isfinite(value).all():
        raise RuntimeError("candidate proposal retention must be None instead of non-finite")
    return float(value.detach().cpu())


def _same_tensor(left: object, right: object) -> bool:
    if (
        type(left) is not torch.Tensor
        or type(right) is not torch.Tensor
        or left.dtype != right.dtype
        or left.device != right.device
        or tuple(left.shape) != tuple(right.shape)
    ):
        return False
    left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return torch.equal(left_bytes, right_bytes)


def _validate_candidate_evidence_types(value: object) -> None:
    """Reject numeric/string subclasses that collide in canonical JSON."""
    if type(value) is torch.Tensor or value is None or type(value) in (bool, int, float, str):
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _validate_candidate_evidence_types(getattr(value, field.name))
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise RuntimeError("candidate proposal diagnostics require built-in string keys")
        for item in value.values():
            _validate_candidate_evidence_types(item)
        return
    if type(value) in (tuple, list):
        for item in value:
            _validate_candidate_evidence_types(item)
        return
    raise RuntimeError(f"candidate proposal evidence contains unsupported type {type(value).__name__}")


def _expected_candidate_retention(
    candidate: CandidateEvaluation,
    *,
    positions_before: torch.Tensor,
    proposed_positions: torch.Tensor,
    pinned: torch.Tensor,
) -> tuple[torch.Tensor | None, bool]:
    free = torch.ones(positions_before.shape[0], dtype=torch.bool, device=positions_before.device)
    free[pinned] = False
    full_displacement = (proposed_positions - positions_before)[free].reshape(-1)
    constrained_displacement = (candidate.constrained_positions - positions_before)[free].reshape(-1)
    full_finite = bool(torch.isfinite(full_displacement).all().item())
    constrained_finite = bool(torch.isfinite(constrained_displacement).all().item())
    if not full_finite or not constrained_finite or torch.equal(full_displacement, torch.zeros_like(full_displacement)):
        retention = None
    else:
        numerator = torch.dot(constrained_displacement, full_displacement)
        denominator = torch.dot(full_displacement, full_displacement)
        if (
            not bool(torch.isfinite(numerator).item())
            or not bool(torch.isfinite(denominator).item())
            or not bool((denominator > 0.0).item())
        ):
            retention = None
        else:
            value = numerator / denominator
            retention = value if bool(torch.isfinite(value).item()) else None
    learned_retained = (
        candidate.step_fraction > 0.0
        and retention is not None
        and bool(torch.isfinite(retention).item())
        and bool((retention > 0.0).item())
    )
    return retention, learned_retained


def _validate_candidate_execution_scalars(config: V5AblationConfig, reference: torch.Tensor) -> None:
    """Mirror core scalar materialization before accepting schema-v4 evidence."""
    for name in (
        "minimum_determinant",
        "minimum_singular_value",
        "proposal_objective_increase_tolerance",
        "proposal_normalized_residual_increase_tolerance",
    ):
        python_value = getattr(config, name)
        materialized = reference.new_tensor(python_value)
        if not bool(torch.isfinite(materialized).item()):
            raise RuntimeError(f"{name} is not finite in the recorded execution dtype")
        if python_value > 0.0 and not bool((materialized > 0.0).item()):
            raise RuntimeError(f"{name} is not positive in the recorded execution dtype")
    safeguard = config.proposal_safeguard
    if type(safeguard) is not ProposalSafeguardConfig:
        raise RuntimeError("schema-v4 evidence lost its proposal safeguard")
    safeguard.validate()
    fractions = reference.new_tensor(safeguard.candidate_step_fractions)
    if not bool(torch.isfinite(fractions).all().item()):
        raise RuntimeError("candidate fractions are not finite in the recorded execution dtype")
    if not bool((fractions[0] == reference.new_tensor(1.0)).item()) or not bool(
        (fractions[-1] == reference.new_tensor(0.0)).item()
    ):
        raise RuntimeError("candidate fraction endpoints changed in the recorded execution dtype")
    if not bool((fractions[:-1] > fractions[1:]).all().item()):
        raise RuntimeError("candidate fractions are not strictly descending in the recorded execution dtype")


def _validate_candidate_tensor_schema(candidate: CandidateEvaluation, reference: torch.Tensor) -> None:
    vector_fields = ("candidate_positions", "constrained_positions", "normalized_residual")
    scalar_fields = (
        "raw_residual_norm",
        "normalized_residual_norm",
        "objective",
        "minimum_determinant",
        "minimum_singular_value",
    )
    for name in vector_fields:
        value = getattr(candidate, name)
        if (
            type(value) is not torch.Tensor
            or value.shape != reference.shape
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            raise RuntimeError("candidate proposal trace changed a vector tensor schema")
    for name in scalar_fields:
        value = getattr(candidate, name)
        if (
            type(value) is not torch.Tensor
            or value.ndim != 0
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            raise RuntimeError("candidate proposal trace changed a scalar tensor schema")


def _validate_iteration_tensor_schema(iteration: IterativeSolverIteration, reference: torch.Tensor) -> None:
    vector_fields = (
        "positions_before",
        "normalized_residual_before",
        "proposed_positions",
        "positions",
        "residual_after",
    )
    scalar_fields = (
        "raw_residual_norm_before",
        "normalized_residual_norm_before",
        "objective_before",
        "minimum_determinant_before",
        "minimum_singular_value_before",
        "raw_residual_norm_after",
        "normalized_residual_norm_after",
        "objective_after",
        "minimum_determinant_after",
        "minimum_singular_value_after",
    )
    for name in vector_fields:
        value = getattr(iteration, name)
        if (
            type(value) is not torch.Tensor
            or value.shape != reference.shape
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            raise RuntimeError("candidate proposal iteration changed a vector tensor schema")
    for name in scalar_fields:
        value = getattr(iteration, name)
        if (
            type(value) is not torch.Tensor
            or value.ndim != 0
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            raise RuntimeError("candidate proposal iteration changed a scalar tensor schema")


def _recomputed_normalized_residual_norm(residual: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(residual.flatten(start_dim=-2), dim=-1)


def _validate_projection_diagnostics(
    diagnostics: torch_solver.ProjectionDiagnostics,
    *,
    expected_backend: str,
) -> None:
    if type(diagnostics) is not torch_solver.ProjectionDiagnostics or diagnostics.backend != expected_backend:
        raise RuntimeError("candidate projection diagnostics changed backend or schema")
    if type(diagnostics.converged) is not bool or type(diagnostics.breakdown) is not bool:
        raise RuntimeError("candidate projection diagnostics changed a decision type")
    integer_fields = (
        "iterations",
        "rhs_count",
        "converged_rhs",
        "matrix_vector_products",
        "preconditioner_applications",
        "factor_solves",
        "hierarchy_levels",
        "preconditioner_matrix_vector_products",
    )
    if any(type(getattr(diagnostics, name)) is not int or getattr(diagnostics, name) < 0 for name in integer_fields):
        raise RuntimeError("candidate projection diagnostics changed a non-negative work count")
    if diagnostics.rhs_count <= 0 or diagnostics.converged_rhs > diagnostics.rhs_count:
        raise RuntimeError("candidate projection diagnostics changed its right-hand-side counts")
    float_fields = (
        "rhs_norm_max",
        "initial_residual_norm_max",
        "residual_norm_max",
        "relative_residual_max",
    )
    if any(
        type(getattr(diagnostics, name)) is not float
        or not math.isfinite(getattr(diagnostics, name))
        or getattr(diagnostics, name) < 0.0
        for name in float_fields
    ):
        raise RuntimeError("candidate projection diagnostics changed a finite residual metric")
    for name in ("relative_tolerance", "absolute_tolerance"):
        value = getattr(diagnostics, name)
        if value is not None and (type(value) is not float or not math.isfinite(value) or value < 0.0):
            raise RuntimeError("candidate projection diagnostics changed a finite tolerance")
    if diagnostics.preconditioner is not None and type(diagnostics.preconditioner) is not str:
        raise RuntimeError("candidate projection diagnostics changed its preconditioner schema")


def _validate_candidate_trace(
    arm: AblationArmResult,
    *,
    safeguard: ProposalSafeguardConfig,
    minimum_determinant: float,
    minimum_singular_value: float,
    objective_increase_tolerance: float,
    normalized_residual_increase_tolerance: float,
    iterations: int,
    persistence: torch.Tensor,
    pinned: torch.Tensor,
    pinned_targets: torch.Tensor,
) -> None:
    """Replay stored candidate decisions under the registered v4 contract.

    This validates arithmetic and gate decisions from the authenticated scored
    tensors. It deliberately does not rerun deformation geometry or the common
    physical objective, because those execution contexts are outside the
    standalone ablation result's return surface.
    """
    trace = arm.proposal_trace
    retention_record = arm.solver_proposal_displacement_retention
    if type(trace) is not tuple or len(trace) != iterations:
        raise RuntimeError("candidate proposal trace changed its fixed K schedule")
    _validate_candidate_evidence_types(trace)
    if type(retention_record) is not tuple or len(retention_record) != iterations:
        raise RuntimeError("candidate proposal retention changed its fixed K schedule")
    fractions = safeguard.candidate_step_fractions
    expected_before = persistence
    previous_iteration: IterativeSolverIteration | None = None
    projection_iterations = 0
    projection_matrix_vector_products = 0
    projection_preconditioner_applications = 0
    projection_factor_solves = 0
    accepted_count = 0
    zero_count = 0
    retained_count = 0
    for iteration_index, iteration in enumerate(trace):
        if type(iteration) is not IterativeSolverIteration:
            raise RuntimeError("candidate proposal trace contains an unregistered iteration record")
        _validate_iteration_tensor_schema(iteration, persistence)
        expected_fraction = iteration_index / max(iterations - 1, 1)
        if (
            type(iteration.iteration) is not int
            or iteration.iteration != iteration_index
            or type(iteration.iteration_fraction) is not float
            or iteration.iteration_fraction != expected_fraction
            or not _same_tensor(iteration.positions_before, expected_before)
        ):
            raise RuntimeError("candidate proposal trace changed its iteration chain")
        if (
            type(iteration.proposed_positions) is not torch.Tensor
            or iteration.proposed_positions.shape != iteration.positions_before.shape
            or iteration.proposed_positions.device != iteration.positions_before.device
            or iteration.proposed_positions.dtype != iteration.positions_before.dtype
            or type(iteration.projection_diagnostics) is not torch_solver.ProjectionDiagnostics
            or iteration.constraint_prepare_diagnostics != {"refreshes": 0}
            or type(iteration.constraint_prepare_diagnostics) is not dict
        ):
            raise RuntimeError("candidate proposal trace changed its registered iteration schema")
        _validate_projection_diagnostics(
            iteration.projection_diagnostics,
            expected_backend=arm.iterative_work.projection_backend,
        )
        projection_iterations += iteration.projection_diagnostics.iterations
        projection_matrix_vector_products += iteration.projection_diagnostics.matrix_vector_products
        projection_preconditioner_applications += iteration.projection_diagnostics.preconditioner_applications
        projection_factor_solves += iteration.projection_diagnostics.factor_solves
        if not _same_tensor(
            iteration.normalized_residual_norm_before,
            _recomputed_normalized_residual_norm(iteration.normalized_residual_before),
        ) or not _same_tensor(
            iteration.normalized_residual_norm_after,
            _recomputed_normalized_residual_norm(iteration.residual_after),
        ):
            raise RuntimeError("candidate iteration normalized residual norm changed after replay")
        if previous_iteration is not None and any(
            not _same_tensor(before, after)
            for before, after in (
                (iteration.normalized_residual_before, previous_iteration.residual_after),
                (iteration.raw_residual_norm_before, previous_iteration.raw_residual_norm_after),
                (iteration.normalized_residual_norm_before, previous_iteration.normalized_residual_norm_after),
                (iteration.objective_before, previous_iteration.objective_after),
                (iteration.minimum_determinant_before, previous_iteration.minimum_determinant_after),
                (iteration.minimum_singular_value_before, previous_iteration.minimum_singular_value_after),
            )
        ):
            raise RuntimeError("candidate proposal trace changed its before/after metric chain")
        candidates = iteration.candidate_evaluations
        if type(candidates) is not tuple or len(candidates) != len(fractions):
            raise RuntimeError("candidate proposal trace changed its fixed candidate schedule")
        for candidate_index, (candidate, step_fraction) in enumerate(zip(candidates, fractions, strict=True)):
            if type(candidate) is not CandidateEvaluation:
                raise RuntimeError("candidate proposal trace contains an unregistered candidate record")
            if (
                type(candidate.candidate_index) is not int
                or candidate.candidate_index != candidate_index
                or type(candidate.step_fraction) is not float
                or candidate.step_fraction != step_fraction
            ):
                raise RuntimeError("candidate proposal trace changed its fixed candidate schedule")
            _validate_candidate_tensor_schema(candidate, iteration.positions_before)
            if not _same_tensor(
                candidate.normalized_residual_norm,
                _recomputed_normalized_residual_norm(candidate.normalized_residual),
            ):
                raise RuntimeError("candidate normalized residual norm changed after replay")
            for name in (
                "positions_finite",
                "exact_pins",
                "determinant_valid",
                "singular_value_valid",
                "objective_finite",
                "residual_finite",
                "state_valid",
                "objective_nonincreasing",
                "residual_nonincreasing",
                "learned_contribution_retained",
                "admissible",
            ):
                if type(getattr(candidate, name)) is not bool:
                    raise RuntimeError("candidate proposal trace changed a canonical decision type")
            if type(candidate.rejection_reasons) is not tuple or any(
                type(reason) is not str for reason in candidate.rejection_reasons
            ):
                raise RuntimeError("candidate proposal trace changed its rejection-reason schema")
            if step_fraction == 0.0:
                if type(candidate.zero_step_unchanged) is not bool:
                    raise RuntimeError("candidate zero-step decision type changed after authentication")
            elif candidate.zero_step_unchanged is not None:
                raise RuntimeError("a positive candidate acquired zero-step metadata")
            if step_fraction == 1.0:
                expected_candidate_positions = iteration.proposed_positions
            elif step_fraction == 0.0:
                expected_candidate_positions = iteration.positions_before
            else:
                expected_candidate_positions = iteration.positions_before + iteration.positions_before.new_tensor(
                    step_fraction
                ) * (iteration.proposed_positions - iteration.positions_before)
            if not _same_tensor(candidate.candidate_positions, expected_candidate_positions):
                raise RuntimeError("candidate interpolation changed after authentication")
            if step_fraction == 0.0 and (
                not _same_tensor(candidate.candidate_positions, iteration.positions_before)
                or not _same_tensor(candidate.constrained_positions, iteration.positions_before)
            ):
                raise RuntimeError("candidate zero candidate changed the exact no-op state")
            if not _same_tensor(candidate.constrained_positions, candidate.candidate_positions):
                raise RuntimeError("candidate identity constraint changed its candidate output")

            expected_zero_unchanged = (
                torch.equal(candidate.constrained_positions, iteration.positions_before)
                if step_fraction == 0.0
                else None
            )
            positions_finite = bool(torch.isfinite(candidate.constrained_positions).all().item())
            exact_pins = torch.equal(candidate.constrained_positions.index_select(0, pinned), pinned_targets)
            determinant_valid = positions_finite and bool(
                torch.isfinite(candidate.minimum_determinant).all().item()
                and (candidate.minimum_determinant > candidate.minimum_determinant.new_tensor(minimum_determinant))
                .all()
                .item()
            )
            singular_value_valid = positions_finite and bool(
                torch.isfinite(candidate.minimum_singular_value).all().item()
                and (
                    candidate.minimum_singular_value
                    > candidate.minimum_singular_value.new_tensor(minimum_singular_value)
                )
                .all()
                .item()
            )
            objective_finite = bool(torch.isfinite(candidate.objective).all().item())
            residual_finite = (
                bool(torch.isfinite(candidate.normalized_residual).all().item())
                and bool(torch.isfinite(candidate.raw_residual_norm).all().item())
                and bool(torch.isfinite(candidate.normalized_residual_norm).all().item())
            )
            state_valid = positions_finite and exact_pins and determinant_valid and singular_value_valid
            objective_nonincreasing = objective_finite and bool(
                (
                    candidate.objective
                    <= iteration.objective_before + candidate.objective.new_tensor(objective_increase_tolerance)
                )
                .all()
                .item()
            )
            residual_nonincreasing = residual_finite and bool(
                (
                    candidate.normalized_residual_norm
                    <= iteration.normalized_residual_norm_before
                    + candidate.normalized_residual_norm.new_tensor(normalized_residual_increase_tolerance)
                )
                .all()
                .item()
            )
            expected_retention, learned_retained = _expected_candidate_retention(
                candidate,
                positions_before=iteration.positions_before,
                proposed_positions=iteration.proposed_positions,
                pinned=pinned,
            )
            if candidate.displacement_retention is None:
                retention_matches = expected_retention is None
            else:
                retention_matches = expected_retention is not None and _same_tensor(
                    candidate.displacement_retention, expected_retention
                )
            rejection_reasons: list[str] = []
            for valid, reason in (
                (positions_finite, "non-finite-positions"),
                (exact_pins, "changed-exact-pins"),
                (determinant_valid, "determinant-bound"),
                (singular_value_valid, "singular-value-bound"),
                (objective_finite, "non-finite-objective"),
                (residual_finite, "non-finite-residual"),
                (objective_nonincreasing, "objective-increase"),
                (residual_nonincreasing, "residual-increase"),
            ):
                if not valid:
                    rejection_reasons.append(reason)
            if expected_zero_unchanged is False:
                rejection_reasons.append("zero-step-moved")
            admissible = (
                state_valid
                and objective_nonincreasing
                and residual_nonincreasing
                and expected_zero_unchanged is not False
            )
            if (
                candidate.positions_finite != positions_finite
                or candidate.exact_pins != exact_pins
                or candidate.determinant_valid != determinant_valid
                or candidate.singular_value_valid != singular_value_valid
                or candidate.objective_finite != objective_finite
                or candidate.residual_finite != residual_finite
                or candidate.state_valid != state_valid
                or candidate.objective_nonincreasing != objective_nonincreasing
                or candidate.residual_nonincreasing != residual_nonincreasing
                or candidate.zero_step_unchanged != expected_zero_unchanged
                or not retention_matches
                or candidate.learned_contribution_retained != learned_retained
                or candidate.admissible != admissible
                or candidate.rejection_reasons != tuple(rejection_reasons)
                or type(candidate.constraint_diagnostics) is not dict
                or candidate.constraint_diagnostics != {"truncation_calls": 0, "minimum_fraction": 1.0}
            ):
                raise RuntimeError("candidate safeguard decisions changed after authenticated replay")

        zero = candidates[-1]
        if not zero.admissible:
            raise RuntimeError("candidate zero candidate is no longer an admissible fallback")
        expected_selection = next(
            (candidate.candidate_index for candidate in candidates[:-1] if candidate.admissible),
            zero.candidate_index,
        )
        if (
            type(iteration.selected_candidate_index) is not int
            or iteration.selected_candidate_index != expected_selection
        ):
            raise RuntimeError("candidate selection changed after authentication")
        selected = candidates[expected_selection]
        if (
            type(iteration.selected_step_fraction) is not float
            or iteration.selected_step_fraction != selected.step_fraction
            or not _same_tensor(iteration.positions, selected.constrained_positions)
            or not _same_tensor(iteration.residual_after, selected.normalized_residual)
            or not _same_tensor(iteration.raw_residual_norm_after, selected.raw_residual_norm)
            or not _same_tensor(iteration.normalized_residual_norm_after, selected.normalized_residual_norm)
            or not _same_tensor(iteration.objective_after, selected.objective)
            or not _same_tensor(iteration.minimum_determinant_after, selected.minimum_determinant)
            or not _same_tensor(iteration.minimum_singular_value_after, selected.minimum_singular_value)
            or type(iteration.constraint_diagnostics) is not dict
            or iteration.constraint_diagnostics != selected.constraint_diagnostics
        ):
            raise RuntimeError("candidate selection no longer matches its committed trace state")
        selected_retention = _proposal_retention_value(selected.displacement_retention)
        iteration_retention = _proposal_retention_value(iteration.learned_displacement_retention)
        if iteration_retention != selected_retention or retention_record[iteration_index] != selected_retention:
            raise RuntimeError("candidate proposal retention changed after authentication")
        if retention_record[iteration_index] is not None and type(retention_record[iteration_index]) is not float:
            raise RuntimeError("candidate proposal retention must use built-in float records")
        proposal_accepted = selected.step_fraction > 0.0
        if (
            type(iteration.proposal_accepted) is not bool
            or iteration.proposal_accepted != proposal_accepted
            or type(iteration.learned_contribution_retained) is not bool
            or iteration.learned_contribution_retained != selected.learned_contribution_retained
        ):
            raise RuntimeError("candidate selection changed its proposal-retention decision")
        if expected_selection == zero.candidate_index:
            expected_reason = "no-admissible-positive"
        elif selected_retention is None:
            expected_reason = "first-admissible-positive-candidate-zero-projected-displacement"
        elif selected.learned_contribution_retained:
            expected_reason = "first-admissible-positive-candidate"
        else:
            expected_reason = "first-admissible-positive-candidate-no-learned-displacement"
        if type(iteration.selection_reason) is not str or iteration.selection_reason != expected_reason:
            raise RuntimeError("candidate selection changed its registered reason")
        accepted_count += int(proposal_accepted)
        zero_count += int(selected.step_fraction == 0.0)
        retained_count += int(selected.learned_contribution_retained)
        expected_before = iteration.positions
        previous_iteration = iteration

    if not _same_tensor(expected_before, arm.pre_corrector_positions):
        raise RuntimeError("candidate proposal trace endpoint differs from the corrector initializer")
    if (
        type(arm.proposal_accepted_iterations) is not int
        or arm.proposal_accepted_iterations != accepted_count
        or type(arm.zero_step_iterations) is not int
        or arm.zero_step_iterations != zero_count
        or type(arm.learned_contribution_retained_iterations) is not int
        or arm.learned_contribution_retained_iterations != retained_count
    ):
        raise RuntimeError("candidate proposal aggregate selection counts changed after authentication")
    work = arm.iterative_work
    if work is None:
        raise RuntimeError("candidate ablation row lost its fixed solver work")
    candidate_count = len(fractions)
    expected_counts = {
        "predictor_passes": iterations,
        "projection_calls": iterations,
        "residual_evaluations": iterations * candidate_count + 1,
        "objective_evaluations": iterations * candidate_count + 1,
        "state_validity_evaluations": iterations * candidate_count + 1,
        "constraint_preparations": iterations,
        "constraint_applications": iterations * candidate_count,
        "physical_step_authentications": iterations * (candidate_count + 1) + 3,
        "common_objective_authentications": iterations * (candidate_count + 1) + 3,
    }
    if any(
        type(getattr(work, name)) is not int or getattr(work, name) != expected
        for name, expected in expected_counts.items()
    ):
        raise RuntimeError("candidate scheduled work changed after authentication")
    if (
        work.projection_diagnostics_recorded is not True
        or type(work.projection_iterations) is not int
        or work.projection_iterations != projection_iterations
        or type(work.projection_matrix_vector_products) is not int
        or work.projection_matrix_vector_products != projection_matrix_vector_products
        or type(work.projection_preconditioner_applications) is not int
        or work.projection_preconditioner_applications != projection_preconditioner_applications
        or type(work.projection_factor_solves) is not int
        or work.projection_factor_solves != projection_factor_solves
    ):
        raise RuntimeError("candidate active projection work differs from its full trace")


def _retention(
    start: torch.Tensor,
    corrected: torch.Tensor,
    persistence: torch.Tensor,
) -> tuple[float, float, float | None]:
    initial = start - persistence
    final = corrected - persistence
    initial_norm = torch.linalg.vector_norm(initial)
    final_norm = torch.linalg.vector_norm(final)
    if bool(initial_norm == 0.0):
        retention = None
    else:
        retention_tensor = (final * initial).sum() / initial.square().sum()
        if not torch.isfinite(retention_tensor):
            raise RuntimeError("initializer-displacement retention is non-finite")
        retention = float(retention_tensor.detach().cpu())
    return float(initial_norm.detach().cpu()), float(final_norm.detach().cpu()), retention


def _canonical_evidence_value(value: object) -> object:
    """Encode every float exactly, including diagnostic NaN/Inf sentinels."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("ablation evidence tensors must have strided layout")
        canonical = value.detach().contiguous()
        raw = canonical.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return {
            "tensor_dtype": str(canonical.dtype),
            "tensor_device": str(value.device),
            "tensor_layout": str(value.layout),
            "tensor_shape": list(canonical.shape),
            "tensor_stride": list(value.stride()),
            "tensor_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_evidence_value(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _canonical_evidence_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_evidence_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, float):
        return _canonical_evidence_value(float(value))
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "+inf" if value > 0.0 else "-inf"}
        return {"float_hex": value.hex()}
    if isinstance(value, str) or value is None:
        return value
    raise TypeError(f"unsupported ablation evidence type {type(value).__name__}")


def _row_evidence_payload(arm: AblationArmResult, schema_version: int) -> dict[str, object]:
    payload = {
        "name": arm.name,
        "start_origin": arm.start_origin,
        "evidence_scope": arm.evidence_scope,
        "vbd_freshness_scope": arm.vbd_freshness_scope,
        "head_mode": arm.head_mode,
        "head_permutation": arm.head_permutation,
        "physical_step_sha256": arm.physical_step_sha256,
        "physical_integration_policy": arm.physical_integration_policy,
        "source_integration_evidence_sha256": arm.source_integration_evidence_sha256,
        "common_objective_sha256": arm.common_objective_sha256,
        "static_mesh_sha256": arm.static_mesh_sha256,
        "operator_geometry_sha256": arm.operator_geometry_sha256,
        "static_graph_sha256": arm.static_graph_sha256,
        "projection_state_sha256": arm.projection_state_sha256,
        "pin_binding_sha256": arm.pin_binding_sha256,
        "predictor_state_sha256": arm.predictor_state_sha256,
        "corrector_config_sha256": arm.corrector_config_sha256,
        "corrector_scheduled_work_sha256": arm.corrector_scheduled_work_sha256,
        "pre_corrector_metrics": dataclasses.asdict(arm.pre_corrector_metrics),
        "corrected_metrics": dataclasses.asdict(arm.corrected_metrics),
        "iterative_work": None if arm.iterative_work is None else dataclasses.asdict(arm.iterative_work),
        "attested_vbd_sweeps": arm.attested_vbd_sweeps,
        "corrector_trace": dataclasses.asdict(arm.corrector_trace),
        "corrector_calls": arm.corrector_calls,
        "correction_norm": arm.correction_norm,
        "correction_rms": arm.correction_rms,
        "start_displacement_from_persistence_norm": arm.start_displacement_from_persistence_norm,
        "corrected_displacement_from_persistence_norm": arm.corrected_displacement_from_persistence_norm,
        "initializer_displacement_retention": arm.initializer_displacement_retention,
        "learned_displacement_retention": arm.learned_displacement_retention,
        "fallback_preserved_start": arm.fallback_preserved_start,
        "pin_overwrite_vertices": arm.pin_overwrite_vertices,
        "vbd_attestation_sha256": arm.vbd_attestation_sha256,
        "vbd_method_record_sha256": arm.vbd_method_record_sha256,
        "timing": dataclasses.asdict(arm.timing),
    }
    if schema_version == 4:
        payload.update(
            {
                "proposal_safeguard_config_sha256": arm.proposal_safeguard_config_sha256,
                "proposal_trace": arm.proposal_trace,
                "solver_proposal_displacement_retention": arm.solver_proposal_displacement_retention,
                "proposal_accepted_iterations": arm.proposal_accepted_iterations,
                "zero_step_iterations": arm.zero_step_iterations,
                "learned_contribution_retained_iterations": arm.learned_contribution_retained_iterations,
            }
        )
    return payload


def _ablation_evidence_sha256(
    *,
    identities: dict[str, object],
    arms: tuple[AblationArmResult, ...],
) -> str:
    schema_version = identities.get("schema_version")
    if schema_version == 3:
        contract = "v5-identical-corrector-development-ablation-v2"
    elif schema_version == 4:
        contract = "v5-identical-corrector-development-ablation-v3"
    else:
        raise ValueError("unsupported ablation evidence schema")
    return canonical_json_sha256(
        _canonical_evidence_value(
            {
                "contract": contract,
                **identities,
                "arms": tuple(_row_evidence_payload(arm, schema_version) for arm in arms),
            }
        )
    )


def _ablation_result_identities(result: V5AblationResult) -> dict[str, object]:
    """Rebuild the complete versioned top-level evidence payload."""
    identities: dict[str, object] = {
        "schema_version": result.schema_version,
        "claim_scope": result.claim_scope,
        "development_only": result.development_only,
        "learned_value_claim": result.learned_value_claim,
        "checkpoint_scope": result.checkpoint_scope,
        "dat_scope": result.dat_scope,
        "physical_step_sha256": result.physical_step_sha256,
        "physical_integration_policy": result.physical_integration_policy,
        "source_integration_evidence_sha256": result.source_integration_evidence_sha256,
        "common_objective_sha256": result.common_objective_sha256,
        "static_mesh_sha256": result.static_mesh_sha256,
        "operator_geometry_sha256": result.operator_geometry_sha256,
        "projection_state_sha256": result.projection_state_sha256,
        "static_graph_sha256": result.static_graph_sha256,
        "predictor_state_sha256": result.predictor_state_sha256,
        "pin_binding_sha256": result.pin_binding_sha256,
        "ablation_config": _ablation_config_payload(result.ablation_config),
        "ablation_config_sha256": result.ablation_config_sha256,
        "corrector_config": dataclasses.asdict(result.corrector_config),
        "corrector_config_sha256": result.corrector_config_sha256,
        "corrector_scheduled_work_sha256": result.corrector_scheduled_work_sha256,
        "learned_scheduled_work_sha256": result.learned_scheduled_work_sha256,
        "head_permutation_sha256": result.head_permutation_sha256,
        "iterations": result.iterations,
        "pinned_vertex_count": result.pinned_vertex_count,
        "corrector_call_count": result.corrector_call_count,
        "vbd_freshness_scope": result.vbd_freshness_scope,
    }
    if result.schema_version == 4:
        identities["proposal_safeguard_config_sha256"] = result.proposal_safeguard_config_sha256
        identities["proposal_pinned_indices"] = result.proposal_pinned_indices
        identities["proposal_pinned_targets"] = result.proposal_pinned_targets
    return identities


def run_v5_identical_corrector_ablation(
    *,
    predictor: StretchPredictor,
    projection_state: SolverState,
    objective: CommonObjectiveContext,
    physical_step: PhysicalStepContext,
    expected_physical_step_sha256: str,
    corrector_config: CorrectorConfig,
    vbd_k1: AttestedVBDK1Start,
    config: V5AblationConfig,
) -> V5AblationResult:
    """Run six starts independently and apply one identical corrector to each.

    The three network arms are fresh calls to the shared architecture-v5
    predictor, including the zero-head control.  Persistence is validated
    against moving pin targets without a silent overwrite.  Only the inertial
    target receives an explicit pin overwrite, which is counted in its row.
    The externally computed VBD row remains caller-attested and unregistered
    in both its row and the returned top-level claim scope.
    """
    if type(config) is not V5AblationConfig:
        raise TypeError("config must be a V5AblationConfig")
    if type(corrector_config) is not CorrectorConfig or type(corrector_config.pcg) is not FixedPCGConfig:
        raise TypeError("corrector_config must be a CorrectorConfig")
    if type(vbd_k1) is not AttestedVBDK1Start:
        raise TypeError("vbd_k1 must be an AttestedVBDK1Start; arbitrary named tensors are not accepted")
    if type(physical_step) is not PhysicalStepContext:
        raise TypeError("physical_step must be a PhysicalStepContext")
    _require_sha256(expected_physical_step_sha256, "expected_physical_step_sha256")
    if predictor_architecture_version(predictor) != 5:
        raise ValueError("identical-corrector ablation requires an architecture-v5 predictor")
    _verify_predictor_execution_surface(predictor)

    physical_step.validate_immutable()
    objective.validate_immutable()
    vbd_k1.validate_immutable()
    _validate_problem_identity(predictor, projection_state, objective)
    _validate_physical_context(projection_state, objective, physical_step)
    if physical_step.physical_step_sha256 != expected_physical_step_sha256:
        raise ValueError("physical-step identity differs from the verified ablation binding")
    x_current, _x_previous, _force, _gravity, _mu, _lam, _pin, pinned_targets_raw = physical_step._owned_tensors()
    if x_current.ndim != 2:
        raise ValueError("the matrix-free corrector ablation requires unbatched physical positions")
    pinned_targets = _expand_pinned_targets(x_current, projection_state.pinned, pinned_targets_raw)
    if not torch.equal(x_current.index_select(0, projection_state.pinned), pinned_targets):
        raise ValueError("persistence must already contain exact current pinned targets")

    permutation = tuple(int(index) for index in config.head_permutation)
    if len(permutation) != projection_state.n_tets or sorted(permutation) != list(range(projection_state.n_tets)):
        raise ValueError("head_permutation must be a bijection over the exact ordered tet count")
    if permutation == tuple(range(projection_state.n_tets)):
        raise ValueError("the permuted-head ablation requires a nonidentity permutation")

    binding_sha256 = pin_binding_sha256(projection_state.pinned, pinned_targets)
    vbd_bindings = {
        "physical_step_sha256": physical_step.physical_step_sha256,
        "common_objective_sha256": objective.common_objective_sha256,
        "static_mesh_sha256": projection_state.static_mesh_sha256,
        "operator_geometry_sha256": projection_state.operator_geometry_sha256,
        "projection_state_sha256": projection_state.projection_state_sha256,
        "pin_binding_sha256": binding_sha256,
    }
    for name, expected in vbd_bindings.items():
        if getattr(vbd_k1, name) != expected:
            raise ValueError(f"VBD-K1 {name} differs from the ablation problem")
    vbd_positions = vbd_k1._owned_positions()
    if vbd_positions.shape != x_current.shape or vbd_positions.device != x_current.device:
        raise ValueError("VBD-K1 positions must match the physical position shape and device")
    if vbd_positions.dtype != x_current.dtype:
        raise ValueError("VBD-K1 positions must match the common-objective execution dtype")
    if not torch.equal(vbd_positions.index_select(0, projection_state.pinned), pinned_targets):
        raise ValueError("VBD-K1 positions must contain the exact physical pinned targets")

    predictor_state_sha256 = learned_state_sha256(predictor.model.state_dict())
    static_graph_sha256 = predictor.model.static_graph_sha256
    candidate_mode = config.proposal_safeguard is not None
    schema_version = 4 if candidate_mode else 3
    ablation_config_sha256 = _ablation_config_sha256(config)
    proposal_safeguard_config_sha256 = (
        None if config.proposal_safeguard is None else _proposal_safeguard_config_sha256(config.proposal_safeguard)
    )
    corrector_config_sha256 = _config_sha256(corrector_config)
    permutation_sha256 = _permutation_sha256(permutation)
    persistence = x_current.clone()
    proposal_pinned_indices = projection_state.pinned.clone() if candidate_mode else None
    proposal_pinned_targets = pinned_targets.clone() if candidate_mode else None

    common_iterative_arguments = {
        "iterations": int(config.iterations),
        "detach_residual_features": config.detach_residual_features,
        "minimum_determinant": float(config.minimum_determinant),
        "minimum_singular_value": float(config.minimum_singular_value),
        "objective_policy": "require-nonincreasing" if candidate_mode else "record",
        "residual_policy": "require-nonincreasing" if candidate_mode else "record",
        "objective_increase_tolerance": config.proposal_objective_increase_tolerance,
        "normalized_residual_increase_tolerance": (config.proposal_normalized_residual_increase_tolerance),
        "return_projection_diagnostics": True,
        "proposal_safeguard": config.proposal_safeguard,
    }
    solve_specs = (
        (LEARNED_ARM, "learned", None),
        (ZERO_ARM, "zero", None),
        (PERMUTED_ARM, "permuted", permutation),
    )
    iterative_results: dict[str, IterativeSolverResult] = {}
    pre_corrector_seconds: dict[str, float] = {}
    with torch.no_grad():
        for arm_name, head_mode, head_permutation in solve_specs:
            _verify_predictor_execution_surface(predictor)
            before = time.perf_counter()
            iterative_config = IterativeSolverConfig(
                **common_iterative_arguments,
                head_mode=head_mode,
                head_permutation=head_permutation,
            )
            result = solve_iterative_principal_stretch(
                predictor=predictor,
                projection_state=projection_state,
                objective=objective,
                physical_step=physical_step,
                expected_physical_step_sha256=expected_physical_step_sha256,
                config=iterative_config,
            )
            pre_corrector_seconds[arm_name] = time.perf_counter() - before
            _validate_iterative_result(
                result,
                head_mode=head_mode,
                head_permutation=head_permutation,
                iterations=int(config.iterations),
                physical_step_sha256=physical_step.physical_step_sha256,
                physical_integration_policy=physical_step.integration_policy,
                source_integration_evidence_sha256=(
                    None if physical_step.source_evidence is None else physical_step.source_evidence.evidence_sha256
                ),
                objective=objective,
                projection_state=projection_state,
                static_graph_sha256=static_graph_sha256,
                persistence=persistence,
            )
            if learned_state_sha256(predictor.model.state_dict()) != predictor_state_sha256:
                raise RuntimeError("predictor state changed while constructing an ablation arm")
            _verify_predictor_execution_surface(predictor)
            iterative_results[arm_name] = result

    learned_scheduled_work_payloads = tuple(
        _iterative_scheduled_work_payload(iterative_results[name].work)
        for name in (LEARNED_ARM, ZERO_ARM, PERMUTED_ARM)
    )
    if not all(payload == learned_scheduled_work_payloads[0] for payload in learned_scheduled_work_payloads[1:]):
        raise RuntimeError("learned, zero, and permuted arms did not schedule identical predictor/projection work")
    learned_scheduled_work_sha256 = canonical_json_sha256(learned_scheduled_work_payloads[0])

    before = time.perf_counter()
    inertial = objective.inertial_target
    if projection_state.pinned.numel() > 0:
        inertial = inertial.index_copy(0, projection_state.pinned, pinned_targets)
    pre_corrector_seconds[INERTIAL_ARM] = time.perf_counter() - before
    pre_corrector_seconds[PERSISTENCE_ARM] = 0.0
    pre_corrector_seconds[VBD_K1_ARM] = 0.0

    starts = {
        LEARNED_ARM: iterative_results[LEARNED_ARM].positions.clone(),
        ZERO_ARM: iterative_results[ZERO_ARM].positions.clone(),
        PERMUTED_ARM: iterative_results[PERMUTED_ARM].positions.clone(),
        PERSISTENCE_ARM: persistence.clone(),
        INERTIAL_ARM: inertial.clone(),
        VBD_K1_ARM: vbd_positions.clone(),
    }
    head_modes: dict[str, str | None] = {
        LEARNED_ARM: "learned",
        ZERO_ARM: "zero",
        PERMUTED_ARM: "permuted",
        PERSISTENCE_ARM: None,
        INERTIAL_ARM: None,
        VBD_K1_ARM: None,
    }

    rows: list[AblationArmResult] = []
    scheduled_work_sha256: str | None = None
    with torch.no_grad():
        for arm_name in MANDATORY_ARM_NAMES:
            physical_step.validate_immutable()
            objective.validate_immutable()
            vbd_k1.validate_immutable()
            _validate_problem_identity(predictor, projection_state, objective)
            if learned_state_sha256(predictor.model.state_dict()) != predictor_state_sha256:
                raise RuntimeError("predictor state changed before identical correction")
            if _ablation_config_sha256(config) != ablation_config_sha256:
                raise RuntimeError("ablation configuration changed after authentication")
            if _config_sha256(corrector_config) != corrector_config_sha256:
                raise RuntimeError("corrector configuration changed after authentication")

            start = starts[arm_name]
            score_before = time.perf_counter()
            start_metrics = _score_endpoint(objective, projection_state, pinned_targets, start)
            start_scoring_seconds = time.perf_counter() - score_before

            correct_before = time.perf_counter()
            correction = correct_common_objective(context=objective, start=start, config=corrector_config)
            corrector_seconds = time.perf_counter() - correct_before
            _validate_corrector_result(
                objective=objective,
                projection_state=projection_state,
                pinned_targets=pinned_targets,
                start=start,
                start_metrics=start_metrics,
                config=corrector_config,
                result_positions=correction.positions,
                direction=correction.direction,
                trace=correction.trace,
            )
            score_after = time.perf_counter()
            corrected_metrics = _score_endpoint(
                objective,
                projection_state,
                pinned_targets,
                correction.positions,
            )
            endpoint_scoring_seconds = time.perf_counter() - score_after
            selected = correction.trace.candidates[correction.trace.selected_candidate_index]
            if correction.trace.accepted:
                if not (
                    _metric_matches(selected.objective, corrected_metrics.total_objective, objective.dtype)
                    and _metric_matches(
                        selected.raw_residual_norm,
                        corrected_metrics.raw_residual_norm,
                        objective.dtype,
                    )
                    and _metric_matches(
                        selected.minimum_determinant,
                        corrected_metrics.minimum_determinant,
                        objective.dtype,
                    )
                    and _metric_matches(
                        selected.minimum_singular_value,
                        corrected_metrics.minimum_singular_value,
                        objective.dtype,
                    )
                ):
                    raise RuntimeError("corrector endpoint differs from independent selected-candidate scoring")
            elif corrected_metrics.positions_sha256 != start_metrics.positions_sha256:
                raise RuntimeError("corrector fallback changed the exact starting endpoint")

            work_sha256 = _corrector_scheduled_work_sha256(correction.trace.work)
            if scheduled_work_sha256 is None:
                scheduled_work_sha256 = work_sha256
            elif work_sha256 != scheduled_work_sha256:
                raise RuntimeError("ablation rows did not execute identical scheduled corrector work")
            correction_delta = correction.positions - start
            correction_norm = float(torch.linalg.vector_norm(correction_delta).detach().cpu())
            correction_rms = float(torch.sqrt(correction_delta.square().mean()).detach().cpu())
            start_norm, corrected_norm, retention = _retention(start, correction.positions, persistence)
            learned_retention = retention if arm_name == LEARNED_ARM else None
            iterative_result = iterative_results.get(arm_name)
            proposal_trace = None
            solver_proposal_retention = None
            proposal_accepted_iterations = None
            zero_step_iterations = None
            learned_contribution_retained_iterations = None
            if candidate_mode and iterative_result is not None:
                proposal_trace = _clone_proposal_trace(iterative_result.trace)
                solver_proposal_retention = tuple(
                    _proposal_retention_value(item.learned_displacement_retention) for item in proposal_trace
                )
                proposal_accepted_iterations = iterative_result.proposal_accepted_iterations
                zero_step_iterations = iterative_result.zero_step_iterations
                learned_contribution_retained_iterations = iterative_result.learned_contribution_retained_iterations
            rows.append(
                AblationArmResult(
                    name=arm_name,
                    start_origin=_ARM_ORIGINS[arm_name],
                    evidence_scope=_CLAIM_SCOPE,
                    vbd_freshness_scope=_VBD_FRESHNESS_SCOPE if arm_name == VBD_K1_ARM else None,
                    head_mode=head_modes[arm_name],
                    head_permutation=permutation if arm_name == PERMUTED_ARM else None,
                    physical_step_sha256=physical_step.physical_step_sha256,
                    physical_integration_policy=physical_step.integration_policy,
                    source_integration_evidence_sha256=(
                        None if physical_step.source_evidence is None else physical_step.source_evidence.evidence_sha256
                    ),
                    common_objective_sha256=objective.common_objective_sha256,
                    static_mesh_sha256=projection_state.static_mesh_sha256,
                    operator_geometry_sha256=projection_state.operator_geometry_sha256,
                    static_graph_sha256=static_graph_sha256,
                    projection_state_sha256=projection_state.projection_state_sha256,
                    pin_binding_sha256=binding_sha256,
                    predictor_state_sha256=predictor_state_sha256,
                    corrector_config_sha256=corrector_config_sha256,
                    corrector_scheduled_work_sha256=work_sha256,
                    vbd_attestation_sha256=vbd_k1.attestation_sha256 if arm_name == VBD_K1_ARM else None,
                    vbd_method_record_sha256=(
                        vbd_k1.method_record.method_record_sha256 if arm_name == VBD_K1_ARM else None
                    ),
                    pre_corrector_positions=start.clone(),
                    corrected_positions=correction.positions.clone(),
                    pre_corrector_metrics=start_metrics,
                    corrected_metrics=corrected_metrics,
                    iterative_work=None if iterative_result is None else iterative_result.work,
                    attested_vbd_sweeps=1 if arm_name == VBD_K1_ARM else 0,
                    corrector_trace=correction.trace,
                    corrector_calls=1,
                    pin_overwrite_vertices=(projection_state.pinned.numel() if arm_name == INERTIAL_ARM else 0),
                    correction_norm=correction_norm,
                    correction_rms=correction_rms,
                    start_displacement_from_persistence_norm=start_norm,
                    corrected_displacement_from_persistence_norm=corrected_norm,
                    initializer_displacement_retention=retention,
                    learned_displacement_retention=learned_retention,
                    fallback_preserved_start=(
                        (not correction.trace.accepted) and torch.equal(correction.positions, start)
                    ),
                    timing=AblationTiming(
                        pre_corrector_seconds=pre_corrector_seconds[arm_name],
                        start_scoring_seconds=start_scoring_seconds,
                        corrector_seconds=corrector_seconds,
                        endpoint_scoring_seconds=endpoint_scoring_seconds,
                    ),
                    proposal_safeguard_config_sha256=(
                        proposal_safeguard_config_sha256 if iterative_result is not None else None
                    ),
                    proposal_trace=proposal_trace,
                    solver_proposal_displacement_retention=solver_proposal_retention,
                    proposal_accepted_iterations=proposal_accepted_iterations,
                    zero_step_iterations=zero_step_iterations,
                    learned_contribution_retained_iterations=(learned_contribution_retained_iterations),
                )
            )

    assert scheduled_work_sha256 is not None
    final_rows = tuple(rows)
    identities = {
        "schema_version": schema_version,
        "claim_scope": _CLAIM_SCOPE,
        "development_only": True,
        "learned_value_claim": False,
        "checkpoint_scope": _CHECKPOINT_SCOPE,
        "dat_scope": _DAT_SCOPE,
        "physical_step_sha256": physical_step.physical_step_sha256,
        "physical_integration_policy": physical_step.integration_policy,
        "source_integration_evidence_sha256": (
            None if physical_step.source_evidence is None else physical_step.source_evidence.evidence_sha256
        ),
        "common_objective_sha256": objective.common_objective_sha256,
        "static_mesh_sha256": projection_state.static_mesh_sha256,
        "operator_geometry_sha256": projection_state.operator_geometry_sha256,
        "projection_state_sha256": projection_state.projection_state_sha256,
        "static_graph_sha256": static_graph_sha256,
        "predictor_state_sha256": predictor_state_sha256,
        "pin_binding_sha256": binding_sha256,
        "ablation_config": _ablation_config_payload(config),
        "ablation_config_sha256": ablation_config_sha256,
        "corrector_config": dataclasses.asdict(corrector_config),
        "corrector_config_sha256": corrector_config_sha256,
        "corrector_scheduled_work_sha256": scheduled_work_sha256,
        "learned_scheduled_work_sha256": learned_scheduled_work_sha256,
        "head_permutation_sha256": permutation_sha256,
        "iterations": int(config.iterations),
        "pinned_vertex_count": projection_state.pinned.numel(),
        "corrector_call_count": len(final_rows),
        "vbd_freshness_scope": _VBD_FRESHNESS_SCOPE,
    }
    if candidate_mode:
        identities["proposal_safeguard_config_sha256"] = proposal_safeguard_config_sha256
        identities["proposal_pinned_indices"] = proposal_pinned_indices
        identities["proposal_pinned_targets"] = proposal_pinned_targets
    evidence_sha256 = _ablation_evidence_sha256(identities=identities, arms=final_rows)
    return V5AblationResult(
        schema_version=schema_version,
        claim_scope=_CLAIM_SCOPE,
        development_only=True,
        learned_value_claim=False,
        checkpoint_scope=_CHECKPOINT_SCOPE,
        dat_scope=_DAT_SCOPE,
        vbd_freshness_scope=_VBD_FRESHNESS_SCOPE,
        physical_step_sha256=physical_step.physical_step_sha256,
        physical_integration_policy=physical_step.integration_policy,
        source_integration_evidence_sha256=(
            None if physical_step.source_evidence is None else physical_step.source_evidence.evidence_sha256
        ),
        common_objective_sha256=objective.common_objective_sha256,
        static_mesh_sha256=projection_state.static_mesh_sha256,
        operator_geometry_sha256=projection_state.operator_geometry_sha256,
        projection_state_sha256=projection_state.projection_state_sha256,
        static_graph_sha256=static_graph_sha256,
        predictor_state_sha256=predictor_state_sha256,
        pin_binding_sha256=binding_sha256,
        ablation_config=config,
        ablation_config_sha256=ablation_config_sha256,
        corrector_config=corrector_config,
        corrector_config_sha256=corrector_config_sha256,
        corrector_scheduled_work_sha256=scheduled_work_sha256,
        learned_scheduled_work_sha256=learned_scheduled_work_sha256,
        head_permutation_sha256=permutation_sha256,
        iterations=int(config.iterations),
        pinned_vertex_count=projection_state.pinned.numel(),
        corrector_call_count=len(final_rows),
        arms=final_rows,
        evidence_sha256=evidence_sha256,
        proposal_safeguard_config_sha256=proposal_safeguard_config_sha256,
        proposal_pinned_indices=proposal_pinned_indices,
        proposal_pinned_targets=proposal_pinned_targets,
    )


def compare_ablation_arms(
    result: V5AblationResult,
    left_arm: str,
    right_arm: str,
) -> AblationComparison:
    """Return numeric endpoint deltas without asserting either arm is better."""
    if type(result) is not V5AblationResult:
        raise TypeError("result must be a V5AblationResult")
    result.validate_immutable()
    if left_arm == right_arm:
        raise ValueError("comparison arms must be distinct")
    left = result.arm(left_arm)
    right = result.arm(right_arm)
    return AblationComparison(
        left_arm=left.name,
        right_arm=right.name,
        start_total_objective_delta=(
            left.pre_corrector_metrics.total_objective - right.pre_corrector_metrics.total_objective
        ),
        corrected_total_objective_delta=(
            left.corrected_metrics.total_objective - right.corrected_metrics.total_objective
        ),
        start_raw_residual_norm_delta=(
            left.pre_corrector_metrics.raw_residual_norm - right.pre_corrector_metrics.raw_residual_norm
        ),
        corrected_raw_residual_norm_delta=(
            left.corrected_metrics.raw_residual_norm - right.corrected_metrics.raw_residual_norm
        ),
        corrected_minimum_determinant_delta=(
            left.corrected_metrics.minimum_determinant - right.corrected_metrics.minimum_determinant
        ),
        corrected_minimum_singular_value_delta=(
            left.corrected_metrics.minimum_singular_value - right.corrected_metrics.minimum_singular_value
        ),
        correction_norm_delta=left.correction_norm - right.correction_norm,
        left_corrector_accepted=left.corrector_trace.accepted,
        right_corrector_accepted=right.corrector_trace.accepted,
        quality_verdict=None,
    )


__all__ = [
    "INERTIAL_ARM",
    "LEARNED_ARM",
    "MANDATORY_ARM_NAMES",
    "PERMUTED_ARM",
    "PERSISTENCE_ARM",
    "VBD_K1_ARM",
    "ZERO_ARM",
    "AblationArmResult",
    "AblationComparison",
    "AblationTiming",
    "AttestedVBDK1Start",
    "EndpointMetrics",
    "V5AblationConfig",
    "V5AblationResult",
    "VBDK1MethodRecord",
    "compare_ablation_arms",
    "pin_binding_sha256",
    "position_sha256",
    "run_v5_identical_corrector_ablation",
]
