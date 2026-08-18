# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Trusted in-memory PR-history loader for v5 training samples.

This module is the narrow trust boundary between an accepted
:class:`PRHistoryChain` transition and the executable v5 trainer.  It accepts
only a canonical root-origin chain whose chain and root hashes were pinned by
the caller.  Raw NumPy content is snapshotted and rehashed before the exact PR
callback scene and common objective are reconstructed.  The stored reference
is then independently rescored and checked against the accepted-reference
policy record.

This is deliberately *not* a durable-artifact loader.  It does not open or
verify a source JSON file, an NPZ bundle, or a :class:`TrajectoryProvenance`.
It also validates the recorded dense-Newton policy and its accepted endpoint;
it does not rerun dense Newton.  A persisted dataset must first verify its
bundle/source bytes and reconstruct this canonical in-memory chain before
calling this module.
"""

from __future__ import annotations

import dataclasses
import json
import types
from collections.abc import Mapping

import numpy as np
import torch

from .correction_ceiling import (
    _chain_member_ordinal,
    _reconstruct_canonical_history,
    _snapshot_chain,
    _validate_accepted_reference_record,
    _verify_history_chain_raw_content,
)
from .iterative_solver import (
    PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
    PhysicalStepContext,
    SolverVBDStagedFloat32Evidence,
    validate_physical_objective_integration,
)
from .pr_scene_history import (
    AtomicCoordinate,
    CommittedState,
    HistoryCheckpoint,
    HistoryTransition,
    PRHistoryChain,
    PRHistoryManifest,
    PRHistoryStaticBundle,
    PRSceneHistory,
    _base_physical_digest,
    _material_digest,
    _root_prefix,
    _topology_digest,
)
from .solver_benchmark import (
    TetBenchmarkScene,
    _array_digest,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
)
from .torch_solver import (
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    build_solver,
    compute_F,
    validate_authenticated_operator_geometry,
)
from .train_pr_history_v5 import V5TrainingSample, canonical_training_tensor_sha256
from .v5_checkpoint import canonical_json_sha256
from .v5_dataset import NumericContentIdentity, TrajectorySampleRecord, canonical_topology_sha256
from .v5_objective import CommonObjectiveContext, common_objective_components, common_objective_residual

_LOADER_SCOPE_PAYLOAD = {
    "schema_version": 2,
    "contract": "pss-v5-pr-history-in-memory-loader-v2",
    "input": "canonical-in-memory-root-origin-PRHistoryChain",
    "external_anchors": ["history-chain-sha256", "root-checkpoint-sha256"],
    "raw_content_authentication": "canonical-snapshot-and-self-hash-recomputation",
    "operator_geometry_policy": OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    "operator_geometry_source": "verified-PR-static-float32-rest-tets-and-tet-poses",
    "execution_dtype": "torch.float64",
    "physical_integration_policy": PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
    "physical_integration_authentication": "exact-source-float32-history-and-staged-SolverVBD-replay",
    "reference_validation": "reconstruct-objective-rescore-endpoint-and-validate-accepted-policy-record",
    "dense_newton_replayed": False,
    "durable_artifact_bundle_opened": False,
    "durable_artifact_source_opened": False,
    "trajectory_provenance_verified": False,
}
LOADER_SCOPE_SHA256 = canonical_json_sha256(_LOADER_SCOPE_PAYLOAD)

_SOURCE_BOUND_LOADER_SCOPE_PAYLOAD = {
    "schema_version": 2,
    "contract": "pss-v5-pr-history-source-bound-loader-v2",
    "input": "externally-anchored-source-bound-PR-history-records-and-exact-arrays",
    "external_anchors": ["history-chain-sha256", "root-checkpoint-sha256"],
    "raw_content_authentication": "canonical-snapshot-and-self-hash-recomputation",
    "base_scene_reconstructed_from_exact_arrays": True,
    "transition_scene_reconstructed_from_source-bound-state": True,
    "source_callback_replayed": False,
    "current_code_reproduction_required": False,
    "operator_geometry_policy": OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    "operator_geometry_source": "verified-source-bound-float32-rest-tets-and-tet-poses",
    "execution_dtype": "torch.float64",
    "physical_integration_policy": PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
    "physical_integration_authentication": "exact-source-float32-history-and-staged-SolverVBD-replay",
    "reference_validation": "reconstruct-objective-rescore-endpoint-and-validate-accepted-policy-record",
    "dense_newton_replayed": False,
    "durable_artifact_bundle_opened": False,
    "durable_artifact_source_opened": False,
    "trajectory_provenance_verified": False,
}
SOURCE_BOUND_LOADER_SCOPE_SHA256 = canonical_json_sha256(_SOURCE_BOUND_LOADER_SCOPE_PAYLOAD)
_LOADER_SCOPES = {
    LOADER_SCOPE_SHA256: _LOADER_SCOPE_PAYLOAD,
    SOURCE_BOUND_LOADER_SCOPE_SHA256: _SOURCE_BOUND_LOADER_SCOPE_PAYLOAD,
}

_MATERIAL_CONTRACT = "pss-v5-runtime-material-mass-mu-lambda-v1"
_PIN_SIGNATURE_CONTRACT = "pss-v5-runtime-pin-factorization-signature-v1"
_REFERENCE_ACCEPTANCE_CONTRACT = "pss-v5-pr-reference-acceptance-binding-v1"
_PHYSICAL_INTEGRATION_BINDING_CONTRACT = "pss-v5-pr-physical-integration-binding-v1"
_LOADED_SAMPLE_CONTRACT = "pss-v5-loaded-pr-history-training-sample-v3"
_REFERENCE_WORK_KEYS = (
    "objective_evaluations",
    "gradient_evaluations",
    "hessian_evaluations",
    "eigenvalue_evaluations",
    "factorization_attempts",
    "line_search_trials",
)


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _freeze_json(value: object, name: str) -> object:
    try:
        canonical = json.loads(json.dumps(_thaw_json(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc

    def freeze(item: object) -> object:
        if isinstance(item, dict):
            return types.MappingProxyType({str(key): freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(canonical)


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_canonical_frozen_json(value: object, name: str) -> None:
    """Reject JSON-compatible aliases that canonical serialization erases."""
    if type(value) is types.MappingProxyType:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{name} contains a noncanonical mapping key")
            _validate_canonical_frozen_json(item, name)
        return
    if type(value) is tuple:
        for item in value:
            _validate_canonical_frozen_json(item, name)
        return
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and np.isfinite(value):
        return
    raise ValueError(f"{name} contains a noncanonical JSON value")


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped values without scalar-subclass or signed-zero aliases."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            matches = [
                index
                for index, (right_key, _) in enumerate(unmatched)
                if type(left_key) is type(right_key) and left_key == right_key
            ]
            if len(matches) != 1:
                return False
            index = matches[0]
            _, right_value = unmatched.pop(index)
            if not _exact_json_equal(left_value, right_value):
                return False
        return not unmatched
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    if left is None or type(left) in (str, int, bool):
        return left == right
    if type(left) is float:
        try:
            return json.dumps(left, allow_nan=False) == json.dumps(right, allow_nan=False)
        except ValueError:
            return False
    return False


def loader_scope(scope_sha256: str = LOADER_SCOPE_SHA256) -> dict[str, object]:
    """Return the authenticated capability and non-capability declaration."""
    if type(scope_sha256) is not str or scope_sha256 not in _LOADER_SCOPES:
        raise ValueError("loader scope SHA-256 is not registered")
    return json.loads(json.dumps(_LOADER_SCOPES[scope_sha256], sort_keys=True))


def canonical_runtime_material_sha256(
    mass: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
) -> str:
    """Hash the exact execution tensors that define v5 material/inertia.

    The identity intentionally differs from the PR manifest's broader material
    identity.  It binds the exact dtype, shape, and bytes consumed by the v5
    common objective: lumped mass plus both per-tet stable-NH coefficients.
    """
    for name, tensor in (("mass", mass), ("mu", mu), ("lam", lam)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be a finite strided floating tensor")
    if mass.ndim != 1 or mu.ndim != 1 or lam.shape != mu.shape:
        raise ValueError("runtime material tensors must have one-dimensional mass and matching one-dimensional mu/lam")
    if mass.device != mu.device or mass.device != lam.device or mass.dtype != mu.dtype or mass.dtype != lam.dtype:
        raise ValueError("runtime material tensors must share device and dtype")
    return canonical_json_sha256(
        {
            "contract": _MATERIAL_CONTRACT,
            "mass_sha256": canonical_training_tensor_sha256(mass),
            "mu_sha256": canonical_training_tensor_sha256(mu),
            "lambda_sha256": canonical_training_tensor_sha256(lam),
        }
    )


def canonical_runtime_pin_signature_sha256(
    pinned: torch.Tensor,
    tets: torch.Tensor,
    n_vertices: int,
) -> str:
    """Hash the exact Dirichlet factorization signature and tet incidence.

    Pin target positions are dynamic physical inputs and are bound by the
    physical-step context.  They are deliberately excluded from this static
    factorization signature.
    """
    if type(n_vertices) is not int or n_vertices < 1:
        raise ValueError("n_vertices must be a positive integer")
    if not isinstance(pinned, torch.Tensor) or not isinstance(tets, torch.Tensor):
        raise TypeError("pinned and tets must be torch.Tensor instances")
    if pinned.dtype != torch.int64 or pinned.ndim != 1 or pinned.layout != torch.strided:
        raise ValueError("pinned must be a one-dimensional strided int64 tensor")
    if tets.dtype != torch.int64 or tets.ndim != 2 or tets.shape[1] != 4 or tets.layout != torch.strided:
        raise ValueError("tets must have shape (T, 4) and strided int64 storage")
    if pinned.device != tets.device:
        raise ValueError("pinned and tets must share one device")
    if pinned.numel():
        if (pinned < 0).any() or (pinned >= n_vertices).any():
            raise ValueError("pinned contains an out-of-range vertex")
        if not torch.equal(pinned, torch.unique(pinned, sorted=True)):
            raise ValueError("pinned must be sorted and unique")
    if (tets < 0).any() or (tets >= n_vertices).any():
        raise ValueError("tets contains an out-of-range vertex")
    incidence = torch.isin(tets, pinned).any(dim=-1)
    return canonical_json_sha256(
        {
            "contract": _PIN_SIGNATURE_CONTRACT,
            "n_vertices": n_vertices,
            "pinned_indices_sha256": canonical_training_tensor_sha256(pinned),
            "tet_pin_incidence_sha256": canonical_training_tensor_sha256(incidence),
        }
    )


def _validated_reference_work(record: Mapping[str, object]) -> Mapping[str, object]:
    work = record.get("work")
    if not isinstance(work, Mapping) or tuple(sorted(work)) != tuple(sorted(_REFERENCE_WORK_KEYS)):
        raise ValueError("accepted-reference record has a noncanonical work-counter schema")
    canonical: dict[str, int] = {}
    for name in _REFERENCE_WORK_KEYS:
        value = work[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"accepted-reference work counter {name} must be a non-negative integer")
        canonical[name] = value
    objective_evaluations = canonical["objective_evaluations"]
    gradient_evaluations = canonical["gradient_evaluations"]
    line_search_trials = canonical["line_search_trials"]
    if objective_evaluations not in (gradient_evaluations, gradient_evaluations + line_search_trials):
        raise ValueError("accepted-reference objective/gradient work counters disagree")
    accepted_iterations = record.get("accepted_iterations")
    if type(accepted_iterations) is not int or accepted_iterations < 0:
        raise ValueError("accepted-reference accepted_iterations must be a non-negative integer")
    if canonical["line_search_trials"] < accepted_iterations:
        raise ValueError("accepted-reference line-search work is below accepted iterations")
    return types.MappingProxyType(canonical)


def _v5_runtime_metrics(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
) -> dict[str, object]:
    """Measure one state in its exact v5 execution dtype and objective."""
    components = common_objective_components(context, positions)
    residual = common_objective_residual(context, positions)
    free = torch.ones(context.n_vertices, dtype=torch.bool, device=context.device)
    free[context.pinned] = False
    raw_residual = float(torch.linalg.vector_norm(residual[free]))
    deformation_gradient = compute_F(positions, context.tets, context.J)
    determinants = torch.linalg.det(deformation_gradient)
    singular_values = torch.linalg.svdvals(deformation_gradient)
    if context.pinned.numel():
        pin_error = torch.linalg.vector_norm(
            positions[context.pinned] - context.inertial_target[context.pinned],
            dim=-1,
        )
        max_pin_error = float(pin_error.max())
    else:
        max_pin_error = 0.0
    return {
        "execution_dtype": str(context.dtype),
        "position_sha256": canonical_training_tensor_sha256(positions),
        "objective_total_joules": float(components["total"]),
        "objective_inertia_joules": float(components["inertia"]),
        "objective_elastic_joules": float(components["elastic"]),
        "raw_free_residual_norm_newtons": raw_residual,
        "residual_scale_newtons": context.residual_scale,
        "normalized_free_residual": raw_residual / context.residual_scale,
        "determinant_min": float(determinants.min()),
        "determinant_max": float(determinants.max()),
        "minimum_singular_value": float(singular_values.min()),
        "max_pin_error_m": max_pin_error,
    }


@dataclasses.dataclass(frozen=True)
class ReferenceAcceptanceBinding:
    """Self-verifying accepted-reference evidence for one loaded sample.

    ``metrics`` are independently recomputed from the reconstructed
    :class:`NewtonProblem`.  ``method``, ``config``, and ``work`` are the
    externally chain-anchored source policy record.  Policy validation proves
    that the endpoint passes the canonical acceptance gates, but
    ``dense_newton_replayed`` remains false: this loader did not reproduce the
    optimization trajectory.
    """

    source_chain_sha256: str
    source_transition_sha256: str
    source_reference_record_sha256: str
    source_scene_sha256: str
    source_objective_instance_sha256: str
    common_objective_sha256: str
    reference_state_sha256: str
    reference_deformation_gradient_sha256: str
    source_reference_positions_sha256: str
    method: str
    config: Mapping[str, object]
    work: Mapping[str, object]
    metrics: Mapping[str, object]
    training_reference_semantics: str
    accepted: bool = True
    policy_record_verified: bool = True
    dense_newton_replayed: bool = False
    committed_image_equilibrium_claimed: bool = False
    acceptance_sha256: str = dataclasses.field(init=False)

    def _validate_scalar_fields(self) -> None:
        for name in (
            "source_chain_sha256",
            "source_transition_sha256",
            "source_reference_record_sha256",
            "source_scene_sha256",
            "source_objective_instance_sha256",
            "common_objective_sha256",
            "reference_state_sha256",
            "reference_deformation_gradient_sha256",
            "source_reference_positions_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.method) is not str or not self.method or self.method != self.method.strip():
            raise ValueError("reference method must be a non-empty canonical string")
        if type(self.training_reference_semantics) is not str or self.training_reference_semantics not in (
            "source-float64-accepted-reference",
            "exact-history-float32-committed-image",
        ):
            raise ValueError("training_reference_semantics is not a registered PR-history bridge")
        if self.accepted is not True or self.policy_record_verified is not True:
            raise ValueError("reference binding requires an accepted, policy-verified endpoint")
        if self.dense_newton_replayed is not False:
            raise ValueError("the in-memory loader must not claim dense-Newton replay")
        if self.committed_image_equilibrium_claimed is not False:
            raise ValueError("the loader must not claim that the float32 committed image is an equilibrium")

    def __post_init__(self) -> None:
        self._validate_scalar_fields()
        for name in ("config", "work", "metrics"):
            object.__setattr__(self, name, _freeze_json(getattr(self, name), f"reference {name}"))
            _validate_canonical_frozen_json(getattr(self, name), f"reference {name}")
        object.__setattr__(self, "acceptance_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _REFERENCE_ACCEPTANCE_CONTRACT,
            "source_chain_sha256": self.source_chain_sha256,
            "source_transition_sha256": self.source_transition_sha256,
            "source_reference_record_sha256": self.source_reference_record_sha256,
            "source_scene_sha256": self.source_scene_sha256,
            "source_objective_instance_sha256": self.source_objective_instance_sha256,
            "common_objective_sha256": self.common_objective_sha256,
            "reference_state_sha256": self.reference_state_sha256,
            "reference_deformation_gradient_sha256": self.reference_deformation_gradient_sha256,
            "source_reference_positions_sha256": self.source_reference_positions_sha256,
            "method": self.method,
            "config": _thaw_json(self.config),
            "work": _thaw_json(self.work),
            "metrics": _thaw_json(self.metrics),
            "training_reference_semantics": self.training_reference_semantics,
            "accepted": self.accepted,
            "policy_record_verified": self.policy_record_verified,
            "dense_newton_replayed": self.dense_newton_replayed,
            "committed_image_equilibrium_claimed": self.committed_image_equilibrium_claimed,
        }

    def validate_immutable(self) -> None:
        """Recompute the binding hash after construction."""
        self._validate_scalar_fields()
        for name in ("config", "work", "metrics"):
            _validate_canonical_frozen_json(getattr(self, name), f"reference {name}")
        _sha256(self.acceptance_sha256, "acceptance_sha256")
        if canonical_json_sha256(self._payload()) != self.acceptance_sha256:
            raise ValueError("reference-acceptance binding changed after authentication")

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible record."""
        payload = self._payload()
        payload["acceptance_sha256"] = self.acceptance_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class PRPhysicalIntegrationBinding:
    """Canonical evidence that one PR source step produced the v5 objective.

    The source tensors remain owned by :class:`SolverVBDStagedFloat32Evidence`.
    This compact record makes their exact identities, the learned
    ``x_previous`` image, and the bound objective target explicit in durable
    metadata without duplicating numeric payloads.
    """

    source_chain_sha256: str
    source_transition_sha256: str
    physical_step_sha256: str
    common_objective_sha256: str
    integration_policy: str
    source_evidence_sha256: str
    dt_seconds: float
    dt_float32_bits: str
    source_pre_event_positions_sha256: str
    source_velocity_sha256: str
    source_mass_sha256: str
    source_inverse_mass_sha256: str
    learned_x_previous_sha256: str
    bound_inertial_target_sha256: str
    binding_sha256: str = dataclasses.field(init=False)

    def _validate_fields(self) -> None:
        for name in (
            "source_chain_sha256",
            "source_transition_sha256",
            "physical_step_sha256",
            "common_objective_sha256",
            "source_evidence_sha256",
            "source_pre_event_positions_sha256",
            "source_velocity_sha256",
            "source_mass_sha256",
            "source_inverse_mass_sha256",
            "learned_x_previous_sha256",
            "bound_inertial_target_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            type(self.integration_policy) is not str
            or self.integration_policy != PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32
        ):
            raise ValueError("PR integration binding requires the staged SolverVBD float32 policy")
        if type(self.dt_seconds) is not float:
            raise TypeError("PR integration dt_seconds must be a canonical float")
        if type(self.dt_float32_bits) is not str:
            raise TypeError("PR integration dt_float32_bits must be a canonical string")
        dt32 = np.float32(self.dt_seconds)
        expected_dt_bits = f"0x{np.asarray(dt32).view(np.uint32).item():08x}"
        if (
            not np.isfinite(dt32)
            or dt32 <= np.float32(0.0)
            or float(dt32) != self.dt_seconds
            or self.dt_float32_bits != expected_dt_bits
        ):
            raise ValueError("PR integration timestep is not its exact authenticated float32 image")

    def __post_init__(self) -> None:
        self._validate_fields()
        object.__setattr__(self, "binding_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _PHYSICAL_INTEGRATION_BINDING_CONTRACT,
            "source_chain_sha256": self.source_chain_sha256,
            "source_transition_sha256": self.source_transition_sha256,
            "physical_step_sha256": self.physical_step_sha256,
            "common_objective_sha256": self.common_objective_sha256,
            "integration_policy": self.integration_policy,
            "source_evidence_sha256": self.source_evidence_sha256,
            "dt_seconds": self.dt_seconds,
            "dt_float32_bits": self.dt_float32_bits,
            "source_pre_event_positions_sha256": self.source_pre_event_positions_sha256,
            "source_velocity_sha256": self.source_velocity_sha256,
            "source_mass_sha256": self.source_mass_sha256,
            "source_inverse_mass_sha256": self.source_inverse_mass_sha256,
            "learned_x_previous_sha256": self.learned_x_previous_sha256,
            "bound_inertial_target_sha256": self.bound_inertial_target_sha256,
        }

    def validate_immutable(self) -> None:
        """Recompute the binding identity after construction."""
        self._validate_fields()
        _sha256(self.binding_sha256, "binding_sha256")
        if canonical_json_sha256(self._payload()) != self.binding_sha256:
            raise ValueError("PR physical-integration binding changed after authentication")

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible integration record."""
        payload = self._payload()
        payload["binding_sha256"] = self.binding_sha256
        return payload


def _sample_source_identity(sample: TrajectorySampleRecord) -> tuple[str, str]:
    source_identity = None
    for name, identity in sample.numeric_content:
        parts = identity.identifier.split(":")
        if len(parts) != 4 or parts[0] != "pr-history-v5" or parts[3] != name:
            raise ValueError(f"sample {name} identifier does not bind canonical PR source provenance")
        chain_sha256 = _sha256(parts[1], f"sample {name} source chain")
        transition_sha256 = _sha256(parts[2], f"sample {name} source transition")
        current = (chain_sha256, transition_sha256)
        if source_identity is None:
            source_identity = current
        elif current != source_identity:
            raise ValueError("sample numeric identities bind inconsistent PR sources")
    if source_identity is None:
        raise ValueError("sample has no numeric source identities")
    return source_identity


def _physical_integration_binding(
    sample: V5TrainingSample,
    source_chain_sha256: str,
    source_transition_sha256: str,
) -> PRPhysicalIntegrationBinding:
    """Recompute the exact PR/VBD evidence record consumed by one sample."""
    step = sample.physical_step
    evidence = step.source_evidence
    if (
        step.integration_policy != PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32
        or type(evidence) is not SolverVBDStagedFloat32Evidence
    ):
        raise ValueError("loaded PR sample is missing canonical staged SolverVBD source evidence")
    evidence.validate_immutable()
    if evidence.source_transition_sha256 != source_transition_sha256:
        raise ValueError("physical integration evidence belongs to a different PR transition")
    if (
        sample.sample_record.physical_integration_policy != step.integration_policy
        or sample.sample_record.source_integration_evidence_sha256 != evidence.evidence_sha256
    ):
        raise ValueError("training sample record differs from its physical integration evidence")
    validate_physical_objective_integration(sample.projection_state, sample.common_objective, step)
    pre_event_positions, velocity, source_mass, inverse_mass = evidence._owned_tensors()
    _, x_previous, *_ = step._owned_tensors()
    inertial_target = sample.common_objective._owned_tensor("inertial_target")
    return PRPhysicalIntegrationBinding(
        source_chain_sha256=source_chain_sha256,
        source_transition_sha256=source_transition_sha256,
        physical_step_sha256=step.physical_step_sha256,
        common_objective_sha256=sample.common_objective.common_objective_sha256,
        integration_policy=step.integration_policy,
        source_evidence_sha256=evidence.evidence_sha256,
        dt_seconds=evidence.dt_seconds,
        dt_float32_bits=evidence.dt_float32_bits,
        source_pre_event_positions_sha256=canonical_training_tensor_sha256(pre_event_positions),
        source_velocity_sha256=canonical_training_tensor_sha256(velocity),
        source_mass_sha256=canonical_training_tensor_sha256(source_mass),
        source_inverse_mass_sha256=canonical_training_tensor_sha256(inverse_mass),
        learned_x_previous_sha256=canonical_training_tensor_sha256(x_previous),
        bound_inertial_target_sha256=canonical_training_tensor_sha256(inertial_target),
    )


@dataclasses.dataclass(frozen=True)
class LoadedPRHistoryV5Sample:
    """One trainer-ready sample plus its independently checked acceptance."""

    training_sample: V5TrainingSample
    reference_acceptance: ReferenceAcceptanceBinding
    physical_integration: PRPhysicalIntegrationBinding
    loader_scope_sha256: str = LOADER_SCOPE_SHA256
    loaded_sample_sha256: str = dataclasses.field(init=False)

    def _validate_field_types(self) -> None:
        if type(self.training_sample) is not V5TrainingSample:
            raise TypeError("training_sample must be a canonical V5TrainingSample")
        if type(self.reference_acceptance) is not ReferenceAcceptanceBinding:
            raise TypeError("reference_acceptance must be a canonical ReferenceAcceptanceBinding")
        if type(self.physical_integration) is not PRPhysicalIntegrationBinding:
            raise TypeError("physical_integration must be a canonical PRPhysicalIntegrationBinding")
        if type(self.loader_scope_sha256) is not str or self.loader_scope_sha256 not in _LOADER_SCOPES:
            raise ValueError("loaded sample changed to an unregistered loader scope")

    def __post_init__(self) -> None:
        self._validate_field_types()
        self.training_sample.validate_immutable()
        self.reference_acceptance.validate_immutable()
        self.physical_integration.validate_immutable()
        self._validate_cross_bindings()
        object.__setattr__(self, "loaded_sample_sha256", canonical_json_sha256(self._payload()))

    def _validate_cross_bindings(self) -> None:
        sample = self.training_sample
        record = sample.sample_record
        acceptance = self.reference_acceptance
        integration = self.physical_integration
        source_chain_sha256, source_transition_sha256 = _sample_source_identity(record)
        if (
            acceptance.source_chain_sha256 != source_chain_sha256
            or acceptance.source_transition_sha256 != source_transition_sha256
        ):
            raise ValueError("reference acceptance belongs to a different PR source than the training sample")
        if acceptance.common_objective_sha256 != record.common_objective_sha256:
            raise ValueError("reference acceptance common objective differs from the training sample")
        if acceptance.reference_state_sha256 != record.reference_state.sha256:
            raise ValueError("reference acceptance state differs from the training sample reference")
        if acceptance.reference_deformation_gradient_sha256 != record.reference_f.sha256:
            raise ValueError("reference acceptance deformation gradient differs from the training sample reference")
        reconstructed_integration = _physical_integration_binding(
            sample,
            source_chain_sha256,
            source_transition_sha256,
        )
        if not _exact_json_equal(reconstructed_integration.as_dict(), integration.as_dict()):
            raise ValueError("physical integration evidence differs from the training sample and PR source")

    def _payload(self) -> dict[str, object]:
        sample = self.training_sample
        source_chain_sha256, source_transition_sha256 = _sample_source_identity(sample.sample_record)
        return {
            "contract": _LOADED_SAMPLE_CONTRACT,
            "loader_scope_sha256": self.loader_scope_sha256,
            "source_chain_sha256": source_chain_sha256,
            "source_transition_sha256": source_transition_sha256,
            "trajectory_id": sample.trajectory_id,
            "sample_sha256": sample.sample_record.sample_sha256,
            "physical_step_sha256": sample.physical_step.physical_step_sha256,
            "common_objective_sha256": sample.common_objective.common_objective_sha256,
            "physical_integration_policy": self.physical_integration.integration_policy,
            "source_integration_evidence_sha256": self.physical_integration.source_evidence_sha256,
            "physical_integration_binding_sha256": self.physical_integration.binding_sha256,
            "projection_state_sha256": sample.projection_state_sha256,
            "reference_acceptance_sha256": self.reference_acceptance.acceptance_sha256,
        }

    def validate_immutable(self) -> None:
        """Reauthenticate the trainer sample and acceptance evidence."""
        self._validate_field_types()
        self.training_sample.validate_immutable()
        self.reference_acceptance.validate_immutable()
        self.physical_integration.validate_immutable()
        self._validate_cross_bindings()
        _sha256(self.loaded_sample_sha256, "loaded_sample_sha256")
        if canonical_json_sha256(self._payload()) != self.loaded_sample_sha256:
            raise ValueError("loaded PR-history sample changed after authentication")

    def as_dict(self) -> dict[str, object]:
        """Return the integrity record without duplicating numeric tensors."""
        payload = self._payload()
        payload["reference_acceptance"] = self.reference_acceptance.as_dict()
        payload["physical_integration"] = self.physical_integration.as_dict()
        payload["loader_scope"] = loader_scope(self.loader_scope_sha256)
        payload["loaded_sample_sha256"] = self.loaded_sample_sha256
        return payload


def _numeric_identity(
    chain_sha256: str,
    transition_sha256: str,
    name: str,
    tensor: torch.Tensor,
) -> NumericContentIdentity:
    return NumericContentIdentity(
        identifier=f"pr-history-v5:{chain_sha256}:{transition_sha256}:{name}",
        sha256=canonical_training_tensor_sha256(tensor),
    )


def _validate_loader_arguments(
    trajectory_id: str,
    expected_history_chain_sha256: str,
    expected_root_checkpoint_sha256: str,
    max_chain_transitions: int,
) -> None:
    if (
        type(trajectory_id) is not str
        or not trajectory_id
        or trajectory_id != trajectory_id.strip()
        or any(character.isspace() for character in trajectory_id)
    ):
        raise ValueError("trajectory_id must be a non-empty canonical string without whitespace")
    _sha256(expected_history_chain_sha256, "expected_history_chain_sha256")
    _sha256(expected_root_checkpoint_sha256, "expected_root_checkpoint_sha256")
    if type(max_chain_transitions) is not int or max_chain_transitions < 1:
        raise ValueError("max_chain_transitions must be a positive integer")


def _source_bound_atomic_scene(
    base_scene: TetBenchmarkScene,
    manifest: PRHistoryManifest,
    transition: HistoryTransition,
) -> TetBenchmarkScene:
    """Rebuild a transition scene without consulting the current checkout."""
    state = transition.input_state
    applied = transition.applied_state
    metadata = dict(base_scene.metadata)
    metadata.update(
        {
            "state_kind": "audited PR callback history atomic substep",
            "history_manifest_sha256": manifest.manifest_sha256,
            "history_state_sha256": state.state_sha256,
            "history_applied_sha256": applied.applied_sha256,
            "history_frame_index": state.coordinate.frame,
            "history_substep_index": state.coordinate.substep,
            "history_callback_action": applied.action,
            "history_callback_applied": applied.callback_applied,
        }
    )
    return dataclasses.replace(
        base_scene,
        name=f"pr2901-{manifest.kind}-history-f{state.coordinate.frame:03d}-s{state.coordinate.substep}",
        x_current=state.q,
        velocity=state.qd,
        particle_flags=applied.particle_flags,
        pinned_indices=applied.pinned_indices,
        pin_targets=applied.pin_targets,
        metadata=metadata,
    )


def _snapshot_source_bound_static(
    manifest: PRHistoryManifest,
    static: PRHistoryStaticBundle,
    base_scene: TetBenchmarkScene,
) -> PRHistoryStaticBundle:
    """Authenticate source static arrays against the reconstructed base scene."""
    if type(manifest) is not PRHistoryManifest or type(static) is not PRHistoryStaticBundle:
        raise ValueError("source-bound loader requires canonical PR manifest/static dataclasses")
    if type(base_scene) is not TetBenchmarkScene or any(callable(value) for value in vars(base_scene).values()):
        raise ValueError("source-bound loader requires a canonical TetBenchmarkScene")
    base_snapshot = dataclasses.replace(base_scene)
    if base_snapshot.manifest() != base_scene.manifest():
        raise ValueError("source-bound base-scene raw content changed after authentication")
    for label, actual, expected in (
        ("base physical", manifest.base_physical_sha256, _base_physical_digest(base_snapshot)),
        ("topology", manifest.topology_sha256, _topology_digest(base_snapshot)),
        ("material", manifest.material_sha256, _material_digest(base_snapshot)),
    ):
        if actual != expected:
            raise ValueError(f"source-bound PR manifest {label} identity differs from the exact base scene")

    static_snapshot = PRHistoryStaticBundle(
        manifest_sha256=static.manifest_sha256,
        base_physical_sha256=static.base_physical_sha256,
        topology_sha256=static.topology_sha256,
        material_sha256=static.material_sha256,
        rest_q=static.rest_q,
        tet_indices=static.tet_indices,
        tet_poses=static.tet_poses,
        mass=static.mass,
        tet_materials=static.tet_materials,
        gravity=static.gravity,
        external_force=static.external_force,
    )
    if static_snapshot.as_dict() != static.as_dict():
        raise ValueError("source-bound PR static raw content changed after authentication")
    expected_static = PRHistoryStaticBundle(
        manifest_sha256=manifest.manifest_sha256,
        base_physical_sha256=manifest.base_physical_sha256,
        topology_sha256=manifest.topology_sha256,
        material_sha256=manifest.material_sha256,
        rest_q=base_snapshot.rest_q,
        tet_indices=base_snapshot.tet_indices,
        tet_poses=base_snapshot.tet_poses,
        mass=base_snapshot.mass,
        tet_materials=base_snapshot.tet_materials,
        gravity=base_snapshot.gravity,
        external_force=base_snapshot.external_force,
    )
    if static_snapshot.as_dict() != expected_static.as_dict():
        raise ValueError("source-bound PR static record differs from the exact base scene")
    for name in ("rest_q", "tet_indices", "tet_poses", "mass", "tet_materials", "gravity", "external_force"):
        if not np.array_equal(getattr(static_snapshot, name), getattr(expected_static, name)):
            raise ValueError(f"source-bound PR static {name} differs from the exact base scene")
    return static_snapshot


def _source_bound_root(manifest: PRHistoryManifest, base_scene: TetBenchmarkScene) -> HistoryCheckpoint:
    state = CommittedState(
        manifest_sha256=manifest.manifest_sha256,
        coordinate=AtomicCoordinate(0, 0),
        q=base_scene.rest_q.astype(np.float32),
        qd=base_scene.velocity.astype(np.float32),
        particle_flags=base_scene.particle_flags,
    )
    return HistoryCheckpoint(
        manifest_sha256=manifest.manifest_sha256,
        state=state,
        prior_transition_sha256=None,
        prefix_sha256=_root_prefix(manifest.manifest_sha256, state.state_sha256),
    )


def _validate_source_bound_commit(transition: HistoryTransition) -> None:
    applied = transition.applied_state
    output = transition.output_state
    expected_q = np.array(transition.reference_positions, dtype=np.float32, order="C", copy=True)
    expected_qd = np.asarray((expected_q - applied.q) / np.float32(transition.dt_seconds), dtype=np.float32)
    if applied.pinned_indices.size:
        expected_qd[applied.pinned_indices] = np.float32(0.0)
    if (
        output.coordinate != transition.next_coordinate
        or not np.array_equal(output.q, expected_q)
        or not np.array_equal(output.qd, expected_qd)
        or not np.array_equal(output.particle_flags, applied.particle_flags)
    ):
        raise ValueError("source-bound transition output differs from the canonical float32 reference commit")
    if not np.array_equal(applied.pin_targets, applied.q[applied.pinned_indices]):
        raise ValueError("source-bound transition pin targets differ from the exact applied state")
    if applied.pinned_indices.size and not np.array_equal(expected_q[applied.pinned_indices], applied.pin_targets):
        raise ValueError("source-bound accepted reference does not preserve exact pin targets")
    if transition.coordinate.substep != 0 and (
        applied.callback_applied
        or applied.action != "none"
        or applied.schedule_value_name is not None
        or applied.schedule_value is not None
        or not np.array_equal(applied.q, transition.input_state.q)
        or not np.array_equal(applied.particle_flags, transition.input_state.particle_flags)
    ):
        raise ValueError("source-bound non-frame transition changed state without a callback")


def _build_loaded_sample_from_verified_scene(
    static: PRHistoryStaticBundle,
    chain_snapshot: PRHistoryChain,
    transition_snapshot: HistoryTransition,
    scene: TetBenchmarkScene,
    *,
    ordinal: int,
    trajectory_id: str,
    device: str | torch.device,
    loader_scope_sha256: str,
) -> LoadedPRHistoryV5Sample:
    if str(scene.manifest()["scene_sha256"]) != transition_snapshot.scene_sha256:
        raise ValueError("reconstructed PR transition scene SHA-256 changed")
    problem = build_common_problem(scene)
    objective_manifest = common_objective_manifest(scene, problem)
    if str(objective_manifest["objective_instance_sha256"]) != transition_snapshot.objective_instance_sha256:
        raise ValueError("reconstructed PR common-objective SHA-256 changed")
    if not np.array_equal(
        problem.inertial_target.detach().numpy(),
        transition_snapshot.inertial_target.astype(np.float64),
    ):
        raise ValueError("reconstructed PR inertial target changed")

    reference_metrics = evaluate_common_state(
        problem,
        transition_snapshot.reference_positions,
        reference_positions=transition_snapshot.reference_positions,
    )
    source_reference_record_sha256 = _validate_accepted_reference_record(
        transition_snapshot,
        problem,
        reference_metrics,
    )
    reference_record = transition_snapshot.reference_record
    reference_work = _validated_reference_work(reference_record)
    reference_config = reference_record.get("config")
    if not isinstance(reference_config, Mapping):
        raise ValueError("accepted-reference record is missing its canonical config")

    target_device = torch.device(device)
    dtype = torch.float64
    pinned_np = np.array(transition_snapshot.applied_state.pinned_indices, dtype=np.int64, copy=True)
    projection_state = build_solver(
        np.array(static.rest_q, copy=True),
        np.array(static.tet_indices, copy=True),
        np.array(static.tet_poses, copy=True),
        pinned_np,
        target_device,
        dtype=dtype,
        tikhonov=0.0,
        projection_backend="dense",
        operator_geometry_policy=OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    )
    operator_geometry_sha256 = validate_authenticated_operator_geometry(projection_state)
    if projection_state.operator_geometry_policy != OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED:
        raise ValueError("dense projection changed the trusted PR source-pose operator policy")
    for name, actual, source, source_dtype in (
        ("rest_q", projection_state.source_rest_q_exact, static.rest_q, torch.float32),
        ("tet_indices", projection_state.source_tet_indices, static.tet_indices, torch.int64),
        ("tet_poses", projection_state.source_tet_poses, static.tet_poses, torch.float32),
    ):
        expected = torch.as_tensor(np.array(source, copy=True), dtype=source_dtype, device=target_device)
        if actual.dtype != source_dtype or not torch.equal(actual, expected):
            raise ValueError(f"dense projection source {name} differs from the verified PR static bundle")

    def floating(value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=target_device, dtype=dtype).clone()
        return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=target_device)

    def source_float32(value: np.ndarray, name: str) -> torch.Tensor:
        raw = np.asarray(value)
        if raw.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"verified PR source {name} must be float32 or an exact float32 promotion")
        source = np.array(raw, dtype=np.float32, order="C", copy=True)
        roundtrip = source if raw.dtype == np.float32 else source.astype(np.float64)
        canonical_raw = np.ascontiguousarray(raw)
        if canonical_raw.shape != roundtrip.shape or canonical_raw.tobytes(order="C") != roundtrip.tobytes(order="C"):
            raise ValueError(f"verified PR source {name} is not an exact float32 image")
        return torch.as_tensor(source, dtype=torch.float32, device=target_device)

    tets = projection_state.tets
    pinned = projection_state.pinned
    mass = floating(problem.mass)
    mu = floating(problem.mu)
    lam = floating(problem.lam)
    inertial_target = floating(problem.inertial_target)
    common_objective = CommonObjectiveContext(
        tets=tets,
        J=projection_state.J,
        volume=projection_state.w,
        mass=mass,
        mu=mu,
        lam=lam,
        inertial_target=inertial_target,
        pinned=pinned,
        dt=transition_snapshot.dt_seconds,
    )

    model_inputs = transition_snapshot.model_inputs()
    x_current = floating(model_inputs["x_current"])
    x_previous = floating(model_inputs["x_previous"])
    force = floating(static.external_force)
    gravity = floating(static.gravity)
    pin = torch.isin(tets, pinned).any(dim=-1).to(dtype=dtype)
    pinned_targets = floating(model_inputs["pin_targets"])
    source_evidence = SolverVBDStagedFloat32Evidence(
        source_transition_sha256=transition_snapshot.transition_sha256,
        dt_seconds=transition_snapshot.dt_seconds,
        pre_event_positions=source_float32(transition_snapshot.input_state.q, "pre-event positions"),
        velocity=source_float32(transition_snapshot.input_state.qd, "velocity"),
        mass=source_float32(static.mass, "mass"),
        inverse_mass=source_float32(scene.particle_inv_mass, "inverse mass"),
    )
    physical_step = PhysicalStepContext(
        x_current=x_current,
        x_previous=x_previous,
        force=force,
        gravity=gravity,
        mu=mu,
        lam=lam,
        pin=pin,
        pinned_targets=pinned_targets,
        integration_policy=PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
        source_evidence=source_evidence,
    )
    validate_physical_objective_integration(projection_state, common_objective, physical_step)

    input_state = floating(transition_snapshot.input_state.q)
    reference_state = floating(transition_snapshot.reference_positions)
    observed_f = compute_F(x_current, tets, projection_state.J)
    input_f = compute_F(input_state, tets, projection_state.J)
    reference_f = compute_F(reference_state, tets, projection_state.J)

    topology_sha256 = canonical_topology_sha256(
        projection_state.source_rest_q.detach().cpu().numpy(),
        tets.detach().cpu().numpy(),
    )
    if topology_sha256 != projection_state.static_mesh_sha256:
        raise ValueError("derived v5 topology disagrees with the dense projection")
    material_sha256 = canonical_runtime_material_sha256(mass, mu, lam)
    pin_signature_sha256 = canonical_runtime_pin_signature_sha256(pinned, tets, projection_state.n_verts)

    tensor_values = {
        "observed_f": observed_f,
        "input_f": input_f,
        "reference_f": reference_f,
        "observed_state": x_current,
        "input_state": input_state,
        "reference_state": reference_state,
    }
    numeric = {
        name: _numeric_identity(chain_snapshot.chain_sha256, transition_snapshot.transition_sha256, name, tensor)
        for name, tensor in tensor_values.items()
    }
    sample_id = f"{trajectory_id}/pr-{ordinal:04d}-{transition_snapshot.transition_sha256[:12]}"
    sample_record = TrajectorySampleRecord(
        sample_id=sample_id,
        ordinal=ordinal,
        topology_sha256=topology_sha256,
        operator_geometry_sha256=operator_geometry_sha256,
        material_sha256=material_sha256,
        pin_signature_sha256=pin_signature_sha256,
        dt_seconds=transition_snapshot.dt_seconds,
        physical_step_sha256=physical_step.physical_step_sha256,
        common_objective_sha256=common_objective.common_objective_sha256,
        physical_integration_policy=physical_step.integration_policy,
        source_integration_evidence_sha256=source_evidence.evidence_sha256,
        observed_f=numeric["observed_f"],
        input_f=numeric["input_f"],
        reference_f=numeric["reference_f"],
        observed_state=numeric["observed_state"],
        input_state=numeric["input_state"],
        reference_state=numeric["reference_state"],
    )
    training_sample = V5TrainingSample(
        trajectory_id=trajectory_id,
        sample_record=sample_record,
        physical_step=physical_step,
        common_objective=common_objective,
        projection_state=projection_state,
        producer_attested_reference_positions=reference_state,
        producer_attested_reference_deformation_gradient=reference_f,
    )
    physical_integration = _physical_integration_binding(
        training_sample,
        chain_snapshot.chain_sha256,
        transition_snapshot.transition_sha256,
    )

    source_position_sha256 = _array_digest(transition_snapshot.reference_positions)
    if source_position_sha256 != reference_metrics.position_sha256:
        raise ValueError("independently measured reference position SHA-256 changed")
    source_committed_metrics = evaluate_common_state(
        problem,
        transition_snapshot.output_state.q.astype(np.float64),
        reference_positions=transition_snapshot.reference_positions,
    )
    committed_runtime_state = floating(transition_snapshot.output_state.q)
    acceptance_metrics = {
        "source_float64_accepted_reference": reference_metrics.as_dict(),
        "source_float64_committed_image": source_committed_metrics.as_dict(),
        "source_objective_committed_minus_accepted_joules": (
            source_committed_metrics.objective - reference_metrics.objective
        ),
        "v5_runtime_training_reference": _v5_runtime_metrics(common_objective, reference_state),
        "v5_runtime_committed_image": _v5_runtime_metrics(common_objective, committed_runtime_state),
    }
    training_reference_semantics = (
        "source-float64-accepted-reference"
        if reference_state.dtype == torch.float64
        else "exact-history-float32-committed-image"
    )
    acceptance = ReferenceAcceptanceBinding(
        source_chain_sha256=chain_snapshot.chain_sha256,
        source_transition_sha256=transition_snapshot.transition_sha256,
        source_reference_record_sha256=source_reference_record_sha256,
        source_scene_sha256=transition_snapshot.scene_sha256,
        source_objective_instance_sha256=transition_snapshot.objective_instance_sha256,
        common_objective_sha256=common_objective.common_objective_sha256,
        reference_state_sha256=numeric["reference_state"].sha256,
        reference_deformation_gradient_sha256=numeric["reference_f"].sha256,
        source_reference_positions_sha256=source_position_sha256,
        method=str(reference_record["method"]),
        config=reference_config,
        work=reference_work,
        metrics=acceptance_metrics,
        training_reference_semantics=training_reference_semantics,
    )
    result = LoadedPRHistoryV5Sample(
        training_sample=training_sample,
        reference_acceptance=acceptance,
        physical_integration=physical_integration,
        loader_scope_sha256=loader_scope_sha256,
    )
    result.validate_immutable()
    return result


def load_pr_history_v5_sample(
    history: PRSceneHistory,
    chain: PRHistoryChain,
    transition: HistoryTransition,
    *,
    trajectory_id: str,
    expected_history_chain_sha256: str,
    expected_root_checkpoint_sha256: str,
    max_chain_transitions: int = 64,
    device: str | torch.device = "cpu",
) -> LoadedPRHistoryV5Sample:
    """Load one accepted PR transition into the executable v5 trainer.

    Args:
        history: Canonical PR scene-history definition owning ``chain``.
        chain: Canonical root-origin accepted chain, held in memory.
        transition: Exact object stored at its global ordinal in ``chain``.
        trajectory_id: Dataset trajectory identifier for the returned sample.
        expected_history_chain_sha256: Externally pinned source-chain digest.
        expected_root_checkpoint_sha256: Externally pinned canonical root
            checkpoint digest.
        max_chain_transitions: Explicit upper bound on raw records traversed.
        device: Torch device for the trainer-ready float64 payload.

    Returns:
        Trainer-ready tensors plus a canonical reference-acceptance binding.

    Raises:
        ValueError: If provenance, raw content, reconstruction, acceptance, or
            any derived v5 identity disagrees.
    """
    _validate_loader_arguments(
        trajectory_id,
        expected_history_chain_sha256,
        expected_root_checkpoint_sha256,
        max_chain_transitions,
    )
    if type(chain) is not PRHistoryChain or type(chain.transitions) is not tuple:
        raise ValueError("loader requires a canonical tuple-backed PRHistoryChain")
    if len(chain.transitions) > max_chain_transitions:
        raise ValueError(
            f"PR history chain has {len(chain.transitions)} transitions, exceeding max_chain_transitions="
            f"{max_chain_transitions}"
        )
    if chain.initial_checkpoint.state.coordinate != AtomicCoordinate(0, 0) or chain.prior_chain_sha256 is not None:
        raise ValueError("loader requires a canonical root-origin PR history chain")

    # Require object identity before copying so an equal-looking transition
    # supplied out of band is never accepted as chain membership.
    ordinal = _chain_member_ordinal(chain, transition)
    chain_snapshot = _snapshot_chain(chain)
    transition_snapshot = chain_snapshot.transitions[ordinal]
    canonical_history = _reconstruct_canonical_history(history)
    _verify_history_chain_raw_content(canonical_history, chain_snapshot)
    PRHistoryChain.verify(chain_snapshot)

    if chain_snapshot.chain_sha256 != expected_history_chain_sha256:
        raise ValueError("PR history chain differs from the externally pinned SHA-256")
    canonical_root = canonical_history.initial_checkpoint
    if canonical_root.checkpoint_sha256 != expected_root_checkpoint_sha256:
        raise ValueError("canonical PR root differs from the externally pinned SHA-256")
    if chain_snapshot.initial_checkpoint.as_dict() != canonical_root.as_dict():
        raise ValueError("PR history chain does not start at the canonical root checkpoint")
    if chain_snapshot.manifest.as_dict() != canonical_history.manifest.as_dict():
        raise ValueError("PR history chain belongs to a different canonical manifest")

    canonical_applied = PRSceneHistory.apply_callback(canonical_history, transition_snapshot.input_state)
    if canonical_applied.as_dict() != transition_snapshot.applied_state.as_dict():
        raise ValueError("transition applied state differs from the canonical PR callback")
    scene = PRSceneHistory.build_atomic_scene(
        canonical_history,
        transition_snapshot.input_state,
        transition_snapshot.applied_state,
    )
    return _build_loaded_sample_from_verified_scene(
        canonical_history.static_bundle,
        chain_snapshot,
        transition_snapshot,
        scene,
        ordinal=ordinal,
        trajectory_id=trajectory_id,
        device=device,
        loader_scope_sha256=LOADER_SCOPE_SHA256,
    )


def load_source_bound_pr_history_v5_sample(
    manifest: PRHistoryManifest,
    static_bundle: PRHistoryStaticBundle,
    base_scene: TetBenchmarkScene,
    chain: PRHistoryChain,
    transition: HistoryTransition,
    *,
    trajectory_id: str,
    expected_history_chain_sha256: str,
    expected_root_checkpoint_sha256: str,
    max_chain_transitions: int = 64,
    device: str | torch.device = "cpu",
) -> LoadedPRHistoryV5Sample:
    """Load an archival sample from externally anchored source-bound content.

    Unlike :func:`load_pr_history_v5_sample`, this entry point does not demand
    that the current checkout reproduce the historical callback manifest.  It
    authenticates the exact persisted base scene, static arrays, root, chain,
    transition commit, objective, and accepted-reference policy record.  A
    durable reader must verify its files and source record before calling this
    function; this function does not itself open artifact files.
    """
    _validate_loader_arguments(
        trajectory_id,
        expected_history_chain_sha256,
        expected_root_checkpoint_sha256,
        max_chain_transitions,
    )
    if type(chain) is not PRHistoryChain or type(chain.transitions) is not tuple:
        raise ValueError("source-bound loader requires a canonical tuple-backed PRHistoryChain")
    if len(chain.transitions) > max_chain_transitions:
        raise ValueError(
            f"PR history chain has {len(chain.transitions)} transitions, exceeding max_chain_transitions="
            f"{max_chain_transitions}"
        )
    if chain.initial_checkpoint.state.coordinate != AtomicCoordinate(0, 0) or chain.prior_chain_sha256 is not None:
        raise ValueError("source-bound loader requires a canonical root-origin PR history chain")
    ordinal = _chain_member_ordinal(chain, transition)
    chain_snapshot = _snapshot_chain(chain)
    transition_snapshot = chain_snapshot.transitions[ordinal]
    PRHistoryChain.verify(chain_snapshot)
    if chain_snapshot.chain_sha256 != expected_history_chain_sha256:
        raise ValueError("source-bound PR history chain differs from the externally pinned SHA-256")
    if chain_snapshot.manifest.as_dict() != manifest.as_dict():
        raise ValueError("source-bound PR history chain belongs to a different manifest")

    base_snapshot = dataclasses.replace(base_scene)
    static_snapshot = _snapshot_source_bound_static(manifest, static_bundle, base_snapshot)
    expected_root = _source_bound_root(manifest, base_snapshot)
    if expected_root.checkpoint_sha256 != expected_root_checkpoint_sha256:
        raise ValueError("source-bound PR root differs from the externally pinned SHA-256")
    if chain_snapshot.initial_checkpoint.as_dict() != expected_root.as_dict():
        raise ValueError("source-bound PR chain does not start at the reconstructed physical root")
    _validate_source_bound_commit(transition_snapshot)
    scene = _source_bound_atomic_scene(base_snapshot, manifest, transition_snapshot)
    return _build_loaded_sample_from_verified_scene(
        static_snapshot,
        chain_snapshot,
        transition_snapshot,
        scene,
        ordinal=ordinal,
        trajectory_id=trajectory_id,
        device=device,
        loader_scope_sha256=SOURCE_BOUND_LOADER_SCOPE_SHA256,
    )
