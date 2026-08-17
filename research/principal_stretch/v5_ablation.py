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
    IterativeSolverConfig,
    IterativeSolverResult,
    IterativeSolverWork,
    PhysicalStepContext,
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

    def __post_init__(self) -> None:
        self.validate_immutable()

    def validate_immutable(self) -> None:
        """Revalidate mutable endpoint bytes and the complete evidence digest."""
        names = tuple(arm.name for arm in self.arms)
        if names != MANDATORY_ARM_NAMES or len(set(names)) != len(names):
            raise ValueError("ablation result must contain each mandatory canonical row exactly once in order")
        if self.schema_version != 2 or self.claim_scope != _CLAIM_SCOPE:
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
        if type(self.ablation_config) is not V5AblationConfig:
            raise TypeError("ablation result must embed the exact V5AblationConfig")
        if type(self.corrector_config) is not CorrectorConfig or type(self.corrector_config.pcg) is not FixedPCGConfig:
            raise TypeError("ablation result must embed the exact CorrectorConfig")
        if _config_sha256(self.ablation_config) != self.ablation_config_sha256:
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
        for arm in self.arms:
            expected_mode, expected_permutation = expected_heads[arm.name]
            if arm.head_mode != expected_mode or arm.head_permutation != expected_permutation:
                raise RuntimeError("ablation row changed its canonical head semantics")
            if arm.evidence_scope != self.claim_scope:
                raise RuntimeError("ablation row changed its development-only evidence scope")
            if arm.start_origin != _ARM_ORIGINS[arm.name]:
                raise RuntimeError("ablation row changed its canonical start origin")
            for name in (
                "physical_step_sha256",
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
            elif arm.iterative_work is not None:
                raise RuntimeError("classical initializer row acquired learned work")
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

        identities = {
            "schema_version": self.schema_version,
            "claim_scope": self.claim_scope,
            "development_only": self.development_only,
            "learned_value_claim": self.learned_value_claim,
            "checkpoint_scope": self.checkpoint_scope,
            "dat_scope": self.dat_scope,
            "physical_step_sha256": self.physical_step_sha256,
            "common_objective_sha256": self.common_objective_sha256,
            "static_mesh_sha256": self.static_mesh_sha256,
            "operator_geometry_sha256": self.operator_geometry_sha256,
            "projection_state_sha256": self.projection_state_sha256,
            "static_graph_sha256": self.static_graph_sha256,
            "predictor_state_sha256": self.predictor_state_sha256,
            "pin_binding_sha256": self.pin_binding_sha256,
            "ablation_config": dataclasses.asdict(self.ablation_config),
            "ablation_config_sha256": self.ablation_config_sha256,
            "corrector_config": dataclasses.asdict(self.corrector_config),
            "corrector_config_sha256": self.corrector_config_sha256,
            "corrector_scheduled_work_sha256": self.corrector_scheduled_work_sha256,
            "learned_scheduled_work_sha256": self.learned_scheduled_work_sha256,
            "head_permutation_sha256": self.head_permutation_sha256,
            "iterations": self.iterations,
            "pinned_vertex_count": self.pinned_vertex_count,
            "corrector_call_count": self.corrector_call_count,
            "vbd_freshness_scope": self.vbd_freshness_scope,
        }
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
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_evidence_value(dataclasses.asdict(value))
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


def _row_evidence_payload(arm: AblationArmResult) -> dict[str, object]:
    return {
        "name": arm.name,
        "start_origin": arm.start_origin,
        "evidence_scope": arm.evidence_scope,
        "vbd_freshness_scope": arm.vbd_freshness_scope,
        "head_mode": arm.head_mode,
        "head_permutation": arm.head_permutation,
        "physical_step_sha256": arm.physical_step_sha256,
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


def _ablation_evidence_sha256(
    *,
    identities: dict[str, object],
    arms: tuple[AblationArmResult, ...],
) -> str:
    return canonical_json_sha256(
        _canonical_evidence_value(
            {
                "contract": "v5-identical-corrector-development-ablation-v2",
                **identities,
                "arms": tuple(_row_evidence_payload(arm) for arm in arms),
            }
        )
    )


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
    ablation_config_sha256 = _config_sha256(config)
    corrector_config_sha256 = _config_sha256(corrector_config)
    permutation_sha256 = _permutation_sha256(permutation)
    persistence = x_current.clone()

    common_iterative_arguments = {
        "iterations": int(config.iterations),
        "detach_residual_features": config.detach_residual_features,
        "minimum_determinant": float(config.minimum_determinant),
        "minimum_singular_value": float(config.minimum_singular_value),
        "objective_policy": "record",
        "residual_policy": "record",
        "return_projection_diagnostics": True,
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
            if _config_sha256(config) != ablation_config_sha256:
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
            rows.append(
                AblationArmResult(
                    name=arm_name,
                    start_origin=_ARM_ORIGINS[arm_name],
                    evidence_scope=_CLAIM_SCOPE,
                    vbd_freshness_scope=_VBD_FRESHNESS_SCOPE if arm_name == VBD_K1_ARM else None,
                    head_mode=head_modes[arm_name],
                    head_permutation=permutation if arm_name == PERMUTED_ARM else None,
                    physical_step_sha256=physical_step.physical_step_sha256,
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
                )
            )

    assert scheduled_work_sha256 is not None
    final_rows = tuple(rows)
    identities = {
        "schema_version": 2,
        "claim_scope": _CLAIM_SCOPE,
        "development_only": True,
        "learned_value_claim": False,
        "checkpoint_scope": _CHECKPOINT_SCOPE,
        "dat_scope": _DAT_SCOPE,
        "physical_step_sha256": physical_step.physical_step_sha256,
        "common_objective_sha256": objective.common_objective_sha256,
        "static_mesh_sha256": projection_state.static_mesh_sha256,
        "operator_geometry_sha256": projection_state.operator_geometry_sha256,
        "projection_state_sha256": projection_state.projection_state_sha256,
        "static_graph_sha256": static_graph_sha256,
        "predictor_state_sha256": predictor_state_sha256,
        "pin_binding_sha256": binding_sha256,
        "ablation_config": dataclasses.asdict(config),
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
    evidence_sha256 = _ablation_evidence_sha256(identities=identities, arms=final_rows)
    return V5AblationResult(
        schema_version=2,
        claim_scope=_CLAIM_SCOPE,
        development_only=True,
        learned_value_claim=False,
        checkpoint_scope=_CHECKPOINT_SCOPE,
        dat_scope=_DAT_SCOPE,
        vbd_freshness_scope=_VBD_FRESHNESS_SCOPE,
        physical_step_sha256=physical_step.physical_step_sha256,
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
