# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Executable authenticated training foundation for principal-stretch v5.

This module consumes already-materialized, self-authenticating tensors.  It is
deliberately not a history-artifact loader: a producer must construct
:class:`V5TrainingSample` values whose runtime contexts and reference tensors
match the frozen :mod:`v5_dataset` records before this trainer can see them.

Topology-specific graph buffers remain separate, while every graph context is
rebound to one canonical set of ``nn.Parameter`` objects.  The optimizer
therefore owns one learned model even when a deterministic sampling schedule
alternates between meshes.

The checkpoint emitted here is the schema-v5 integrity-only continuation
snapshot.  It is not an exact-resume proof.  Inference safeguards remain the
strict policies authenticated by :class:`V5SolverContract`; the finite,
orientation-preserving training rollout uses an explicit separately hashed
development policy because an untrained model cannot in general satisfy
monotone objective and residual gates.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections.abc import Mapping, Sequence

import torch

from .graph_transformer import GraphTransformerConfig
from .iterative_solver import (
    IdentityConstraintHook,
    IterativeSolverConfig,
    IterativeSolverResult,
    PhysicalStepContext,
    solve_iterative_principal_stretch,
)
from .predictor import StretchPredictor, build_stretch_predictor
from .torch_solver import (
    SolverState,
    compute_F,
    projection_state_sha256,
    validate_authenticated_operator_geometry,
)
from .v5_checkpoint import (
    OptimizerContract,
    ParentLineage,
    ProjectionContract,
    TrainingStage,
    V5SolverContract,
    VerifiedV5Checkpoint,
    _verify_projection_operator,
    build_v5_checkpoint,
    canonical_json_sha256,
    verify_v5_checkpoint,
)
from .v5_dataset import (
    DataAccessLedger,
    DataAccessPurpose,
    DataAccessScope,
    DatasetRole,
    SamplingBatch,
    SamplingReference,
    SamplingSchedule,
    TrajectorySampleRecord,
    build_sampling_schedule,
    canonical_topology_sha256,
)
from .v5_objective import CommonObjectiveContext
from .v5_training import (
    CompatibleStateLoss,
    PotentialExcessLoss,
    RepresentationLoss,
    build_principal_stretch_labels,
    common_potential_excess_loss_batch,
    compatible_state_loss,
    principal_stretch_representation_loss,
)

_TRAINING_TENSOR_CONTRACT = b"pss-v5-training-tensor-v1\0"
_OPENED_PAYLOAD_NAMES = ("common_objective", "physical_step", "reference_f", "reference_state")
_TRAINER_EXECUTION_PAYLOAD = {
    "schema_version": 1,
    "contract": "pss-v5-executable-training-foundation-v1",
    "representation_reduction": "arithmetic-mean-over-samples-and-K-recurrent-updates",
    "physics_reduction": "mean-compatible-state-total-plus-per-sample-common-potential-excess-batch-total",
    "physics_baseline": "authenticated-physical-persistence-x_current-per-sample",
    "physical_objective_routing": "one-authenticated-CommonObjectiveContext-per-sample",
    "history_policy": "x_current-and-x_previous-fixed-x_iterate-recurrent",
    "projection": "dense-differentiable-full-deformation-gradient",
    "constraint": "exact-registered-identity",
    "training_objective_policy": "record",
    "training_residual_policy": "record",
    "training_state_acceptance": "finite-exact-pins-strict-contract-determinant-and-singular-value-bounds",
    "label_rejection": "fail-closed-no-clipping-or-sample-dropping",
    "reference_semantics": "producer-attested-supervision-no-equilibrium-or-acceptance-proof",
    "gradient_reduction": "one-optimizer-update-per-SamplingBatch",
    "stage_boundary_optimizer": "single-AdamW-state-continues-without-reset",
    "head_freezing": "none-all-learned-parameters-train-in-both-stages",
    "trajectory_supervision": "teacher-forced-per-sample-physical-history-no-rollout-sampling",
    "checkpoint_binding": "integrity-snapshot-only-not-V5SolverContract-training-semantics",
    "resume_capability": False,
}
TRAINER_EXECUTION_CONTRACT_SHA256 = canonical_json_sha256(_TRAINER_EXECUTION_PAYLOAD)

_ADAMW_KEYS = {
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "beta1",
    "beta2",
    "epsilon",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
}
_LEARNED_DTYPES = {"torch.float32": torch.float32, "torch.float64": torch.float64}


def _canonical_tensor_bytes(tensor: torch.Tensor) -> tuple[bytes, bytes]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("authenticated training values must be torch.Tensor instances")
    if tensor.layout != torch.strided:
        raise ValueError("authenticated training tensors must have strided layout")
    value = tensor.detach().contiguous()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("authenticated training tensors must be finite")
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    raw = value.view(torch.uint8).cpu().numpy().tobytes()
    return metadata, raw


def canonical_training_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one materialized numeric tensor for a trainer-ready dataset record."""
    metadata, raw = _canonical_tensor_bytes(tensor)
    digest = hashlib.sha256(_TRAINING_TENSOR_CONTRACT)
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def trainer_execution_contract() -> dict[str, object]:
    """Return a copy of the separately authenticated training semantics."""
    return json.loads(json.dumps(_TRAINER_EXECUTION_PAYLOAD, sort_keys=True))


def build_v5_adamw_optimizer_contract(
    *,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> OptimizerContract:
    """Build the only optimizer policy executed by this foundation trainer."""
    return OptimizerContract.build(
        "AdamW",
        {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        },
    )


def _verified_record(record: TrajectorySampleRecord) -> None:
    if type(record) is not TrajectorySampleRecord:
        raise TypeError("sample_record must be a canonical TrajectorySampleRecord")
    payload = record.as_dict()
    declared = payload.pop("sample_sha256")
    if declared != record.sample_sha256 or canonical_json_sha256(payload) != declared:
        raise ValueError("training sample record changed after authentication")


def _same_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.device != expected.device or actual.dtype != expected.dtype or not torch.equal(actual, expected):
        raise ValueError(f"training payload {name} differs from its bound runtime context")


@dataclasses.dataclass(frozen=True)
class V5TrainingSample:
    """One sealed, trainer-ready sample with explicit runtime payloads.

    The reference tensors are producer-attested supervision, not proof of an
    equilibrium, solver acceptance gate, or promotion-quality endpoint. A
    future real-artifact loader must require a canonical per-sample acceptance
    record binding method/config/work and residual/objective/validity metrics
    before making any such claim. ``reference_state.sha256`` and
    ``reference_f.sha256`` in ``sample_record`` must use
    :func:`canonical_training_tensor_sha256`. The remaining numeric
    artifacts are not opened by this trainer; their durable-file verification
    remains the responsibility of the producer/loader that created the frozen
    manifest.  In particular, ``material_sha256`` and
    ``pin_signature_sha256`` remain opaque manifest/schedule declarations:
    this module checks their equality along that chain and checks actual
    material/pin tensors through the physical/objective digests, but the
    dataset foundation does not yet define a canonical runtime builder that
    equates those two declaration hashes with the tensors.
    """

    trajectory_id: str
    sample_record: TrajectorySampleRecord
    physical_step: PhysicalStepContext
    common_objective: CommonObjectiveContext
    projection_state: SolverState
    producer_attested_reference_positions: torch.Tensor
    producer_attested_reference_deformation_gradient: torch.Tensor
    producer_attested_reference_positions_sha256: str = dataclasses.field(init=False)
    producer_attested_reference_deformation_gradient_sha256: str = dataclasses.field(init=False)
    operator_geometry_sha256: str = dataclasses.field(init=False)
    projection_state_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.trajectory_id) is not str
            or not self.trajectory_id
            or self.trajectory_id != self.trajectory_id.strip()
        ):
            raise ValueError("trajectory_id must be a non-empty canonical string")
        _verified_record(self.sample_record)
        if type(self.physical_step) is not PhysicalStepContext:
            raise TypeError("physical_step must be a canonical PhysicalStepContext")
        if type(self.common_objective) is not CommonObjectiveContext:
            raise TypeError("common_objective must be a canonical CommonObjectiveContext")
        if type(self.projection_state) is not SolverState:
            raise TypeError("projection_state must be a SolverState")
        if not isinstance(self.producer_attested_reference_positions, torch.Tensor) or not isinstance(
            self.producer_attested_reference_deformation_gradient, torch.Tensor
        ):
            raise TypeError("reference positions and deformation gradients must be tensors")

        object.__setattr__(
            self,
            "producer_attested_reference_positions",
            self.producer_attested_reference_positions.detach().clone(),
        )
        object.__setattr__(
            self,
            "producer_attested_reference_deformation_gradient",
            self.producer_attested_reference_deformation_gradient.detach().clone(),
        )
        positions_sha256 = canonical_training_tensor_sha256(self.producer_attested_reference_positions)
        deformation_sha256 = canonical_training_tensor_sha256(self.producer_attested_reference_deformation_gradient)
        object.__setattr__(self, "producer_attested_reference_positions_sha256", positions_sha256)
        object.__setattr__(
            self,
            "producer_attested_reference_deformation_gradient_sha256",
            deformation_sha256,
        )
        operator_sha256 = validate_authenticated_operator_geometry(self.projection_state)
        object.__setattr__(self, "operator_geometry_sha256", operator_sha256)
        object.__setattr__(self, "projection_state_sha256", projection_state_sha256(self.projection_state))
        self.validate_immutable()

    @property
    def key(self) -> tuple[str, str]:
        """Return the schedule lookup key."""
        return self.trajectory_id, self.sample_record.sample_id

    def validate_immutable(self) -> None:
        """Reauthenticate every consumed payload before learned execution."""
        _verified_record(self.sample_record)
        self.physical_step.validate_immutable()
        self.common_objective.validate_immutable()
        if self.sample_record.physical_step_sha256 != self.physical_step.physical_step_sha256:
            raise ValueError("physical-step SHA-256 differs from the training sample record")
        if self.sample_record.common_objective_sha256 != self.common_objective.common_objective_sha256:
            raise ValueError("common-objective SHA-256 differs from the training sample record")
        if (
            canonical_training_tensor_sha256(self.producer_attested_reference_positions)
            != self.producer_attested_reference_positions_sha256
        ):
            raise ValueError("reference positions changed after authentication")
        if (
            canonical_training_tensor_sha256(self.producer_attested_reference_deformation_gradient)
            != self.producer_attested_reference_deformation_gradient_sha256
        ):
            raise ValueError("reference deformation gradient changed after authentication")
        if self.producer_attested_reference_positions_sha256 != self.sample_record.reference_state.sha256:
            raise ValueError("reference positions SHA-256 differs from reference_state in the sample record")
        if self.producer_attested_reference_deformation_gradient_sha256 != self.sample_record.reference_f.sha256:
            raise ValueError("reference deformation-gradient SHA-256 differs from reference_f in the sample record")

        state = self.projection_state
        actual_operator_sha256 = validate_authenticated_operator_geometry(state)
        if (
            actual_operator_sha256 != self.operator_geometry_sha256
            or actual_operator_sha256 != self.sample_record.operator_geometry_sha256
        ):
            raise ValueError("operator-geometry SHA-256 differs from the training sample record")
        actual_projection_sha256 = projection_state_sha256(state)
        if (
            state.projection_state_sha256 != actual_projection_sha256
            or actual_projection_sha256 != self.projection_state_sha256
        ):
            raise ValueError("dense projection state changed after training-payload authentication")
        if state.projection_backend != "dense" or state.tikhonov != 0.0:
            raise ValueError("v5 training requires exact dense differentiable projection without tikhonov")
        _verify_projection_operator(
            state,
            ProjectionContract(
                backend="dense",
                relative_tolerance=None,
                absolute_tolerance=None,
                max_iterations=0,
                warm_start="not-applicable",
                raise_on_nonconvergence=True,
                preconditioner="none",
                require_runtime_diagnostics=True,
                execution_dtype=str(state.rest_q.dtype),
                operator_geometry_policy=state.operator_geometry_policy,
            ),
        )
        if state.source_rest_q.dtype != torch.float64:
            raise ValueError("projection source_rest_q must preserve canonical float64 source geometry")
        _verify_sample_topology(self)

        objective = self.common_objective
        if state.n_verts != objective.n_vertices or state.n_tets != objective.n_tets:
            raise ValueError("projection and common-objective mesh sizes differ")
        if state.rest_q.device != objective.device or state.rest_q.dtype != objective.dtype:
            raise ValueError("projection and common objective must share device and floating dtype")
        for name, projected, bound in (
            ("tets", state.tets, objective.tets),
            ("J", state.J, objective.J),
            ("volume", state.w, objective.volume),
            ("pinned", state.pinned, objective.pinned),
        ):
            _same_tensor(name, projected, bound)
        for name in ("tets", "J", "volume", "mass", "mu", "lam", "inertial_target", "pinned"):
            if objective._owned_tensor(name).requires_grad:
                raise ValueError(f"training common-objective tensor {name} must not require gradients")
        if objective.dt != self.sample_record.dt_seconds:
            raise ValueError("common-objective timestep differs from the sample record")

        x_current, x_previous, force, gravity, mu, lam, pin, pinned_targets = self.physical_step._owned_tensors()
        for name, value in (
            ("x_current", x_current),
            ("x_previous", x_previous),
            ("force", force),
            ("gravity", gravity),
            ("mu", mu),
            ("lam", lam),
            ("pin", pin),
            ("pinned_targets", pinned_targets),
            ("producer_attested_reference_positions", self.producer_attested_reference_positions),
            (
                "producer_attested_reference_deformation_gradient",
                self.producer_attested_reference_deformation_gradient,
            ),
        ):
            if value.device != objective.device or value.dtype != objective.dtype:
                raise ValueError(f"{name} must share the common objective device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            if value.requires_grad:
                raise ValueError(f"training data tensor {name} must not require gradients")
        for name in (
            "Dm_inv",
            "J",
            "w",
            "L",
            "L_ff_chol",
            "L_fp",
            "rest_q",
            "source_rest_q",
            "source_rest_q_exact",
            "source_tet_indices",
            "source_tet_poses",
        ):
            value = getattr(state, name)
            if value is not None and value.requires_grad:
                raise ValueError(f"training projection tensor {name} must not require gradients")
        if x_current.ndim != 2:
            raise ValueError("V5TrainingSample represents exactly one unbatched physical transition")
        if self.producer_attested_reference_positions.shape != (state.n_verts, 3):
            raise ValueError("producer-attested reference positions have the wrong mesh shape")
        if self.producer_attested_reference_deformation_gradient.shape != (state.n_tets, 3, 3):
            raise ValueError("producer-attested reference deformation gradient has the wrong mesh shape")
        _same_tensor("mu", mu, objective.mu)
        _same_tensor("lam", lam, objective.lam)
        expected_pin = torch.isin(state.tets, state.pinned).any(dim=-1).to(pin)
        _same_tensor("pin incidence", pin, expected_pin)
        if not torch.equal(x_current[state.pinned], pinned_targets):
            raise ValueError("physical current state does not contain exact pinned targets")
        if not torch.equal(self.producer_attested_reference_positions[state.pinned], pinned_targets):
            raise ValueError("producer-attested reference does not preserve exact pinned targets")
        expected_reference_f = compute_F(self.producer_attested_reference_positions, state.tets, state.J)
        _same_tensor(
            "producer-attested reference deformation gradient",
            self.producer_attested_reference_deformation_gradient,
            expected_reference_f,
        )
        if (torch.linalg.det(self.producer_attested_reference_deformation_gradient) <= 0.0).any():
            raise ValueError("producer-attested reference deformation must have positive orientation")

        free_mask = torch.ones(state.n_verts, dtype=torch.bool, device=objective.device)
        free_mask[state.pinned] = False
        mass = objective._owned_tensor("mass")
        inertial_target = objective._owned_tensor("inertial_target")
        acceleration = gravity[None, :] + force[free_mask] / mass[free_mask, None]
        expected_target = 2.0 * x_current[free_mask] - x_previous[free_mask]
        expected_target = expected_target + objective.dt * objective.dt * acceleration
        if not torch.allclose(expected_target, inertial_target[free_mask], rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("physical history and loads differ from the bound common objective")


def _verify_sample_topology(sample: V5TrainingSample) -> str:
    state = sample.projection_state
    actual = canonical_topology_sha256(
        state.source_rest_q.detach().cpu().numpy(),
        state.tets.detach().cpu().numpy(),
    )
    if actual != sample.sample_record.topology_sha256 or actual != state.static_mesh_sha256:
        raise ValueError("materialized projection topology differs from the training sample record")
    if validate_authenticated_operator_geometry(state) != sample.sample_record.operator_geometry_sha256:
        raise ValueError("materialized projection operator differs from the training sample record")
    return actual


@dataclasses.dataclass(frozen=True)
class V5BatchLoss:
    """One exact scheduled-batch scalar and its registered components."""

    total: torch.Tensor
    representation: torch.Tensor | None
    compatible_state: torch.Tensor | None
    common_potential_excess: torch.Tensor | None
    final_positions: tuple[torch.Tensor, ...]


def _module_for_parameter(model: torch.nn.Module, parameter_name: str) -> tuple[torch.nn.Module, str]:
    pieces = parameter_name.split(".")
    module = model
    for piece in pieces[:-1]:
        module = module._modules[piece]
    return module, pieces[-1]


class SharedTopologyPredictorBank:
    """Topology-specific static graphs backed by one learned Parameter set."""

    def __init__(self, graph_config: GraphTransformerConfig, learned_dtype: torch.dtype):
        if type(graph_config) is not GraphTransformerConfig or graph_config.architecture_version != 5:
            raise ValueError("shared predictor bank requires canonical architecture-v5 graph config")
        if learned_dtype not in (torch.float32, torch.float64):
            raise ValueError("shared predictor bank requires float32 or float64 learned parameters")
        self.graph_config = graph_config
        self.learned_dtype = learned_dtype
        self._predictors: dict[str, StretchPredictor] = {}
        self._master_topology_sha256: str | None = None
        self._parameter_names: tuple[str, ...] = ()

    def ensure(self, sample: V5TrainingSample) -> StretchPredictor:
        """Return a context, constructing and parameter-tying it if needed."""
        sample.validate_immutable()
        topology = _verify_sample_topology(sample)
        existing = self._predictors.get(topology)
        if existing is not None:
            self._validate_predictor(existing, sample)
            return existing
        state = sample.projection_state
        predictor = build_stretch_predictor(
            "graph-transformer",
            state.source_rest_q.detach().cpu().numpy(),
            state.tets.detach().cpu().numpy(),
            state.rest_q.device,
            self.learned_dtype,
            residual=True,
            graph_config=self.graph_config,
        )
        if self._master_topology_sha256 is None:
            self._master_topology_sha256 = topology
            self._parameter_names = tuple(name for name, _parameter in predictor.model.named_parameters())
        else:
            master = self.master
            master_parameters = dict(master.model.named_parameters())
            candidate_parameters = dict(predictor.model.named_parameters())
            if (
                tuple(candidate_parameters) != self._parameter_names
                or tuple(master_parameters) != self._parameter_names
            ):
                raise ValueError("topology graph contexts do not share one exact learned parameter schema")
            for name in self._parameter_names:
                candidate = candidate_parameters[name]
                shared = master_parameters[name]
                if (
                    candidate.shape != shared.shape
                    or candidate.dtype != shared.dtype
                    or candidate.device != shared.device
                ):
                    raise ValueError("topology graph context has an incompatible learned parameter")
                module, leaf = _module_for_parameter(predictor.model, name)
                module._parameters[leaf] = shared
        self._predictors[topology] = predictor
        self._validate_predictor(predictor, sample)
        self._validate_shared_parameter_identity()
        return predictor

    @property
    def master(self) -> StretchPredictor:
        """Return the predictor that owns the canonical checkpoint state."""
        if self._master_topology_sha256 is None:
            raise RuntimeError("predictor bank has no topology context")
        return self._predictors[self._master_topology_sha256]

    @property
    def predictors(self) -> Mapping[str, StretchPredictor]:
        """Return a shallow topology-to-context view."""
        return dict(self._predictors)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Canonical optimizer/checkpoint parameter order."""
        return self._parameter_names

    def shared_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """Return each unique learned Parameter exactly once and in name order."""
        parameters = dict(self.master.model.named_parameters())
        result = tuple(parameters[name] for name in self._parameter_names)
        if len({id(parameter) for parameter in result}) != len(result):
            raise RuntimeError("master learned parameter schema unexpectedly aliases distinct names")
        return result

    def _validate_predictor(self, predictor: StretchPredictor, sample: V5TrainingSample) -> None:
        state = sample.projection_state
        if predictor.model.static_mesh_sha256 != state.static_mesh_sha256:
            raise ValueError("topology predictor and projection static mesh differ")
        if predictor.model.static_graph_sha256 != predictor.model.compute_static_graph_sha256():
            raise ValueError("topology predictor static graph changed after construction")
        if not torch.equal(predictor.model.tets, state.tets):
            raise ValueError("topology predictor and projection ordered tetrahedra differ")

    def _validate_shared_parameter_identity(self) -> None:
        if not self._predictors:
            return
        master = dict(self.master.model.named_parameters())
        for predictor in self._predictors.values():
            candidate = dict(predictor.model.named_parameters())
            if tuple(candidate) != self._parameter_names:
                raise RuntimeError("topology predictor parameter names changed")
            if any(candidate[name] is not master[name] for name in self._parameter_names):
                raise RuntimeError("topology predictors do not share exact Parameter objects")


def _reference_matches(reference: SamplingReference, sample: V5TrainingSample) -> None:
    record = sample.sample_record
    expected = {
        "trajectory_id": sample.trajectory_id,
        "topology_sha256": record.topology_sha256,
        "operator_geometry_sha256": record.operator_geometry_sha256,
        "pin_signature_sha256": record.pin_signature_sha256,
        "static_layout_sha256": record.static_layout_sha256,
        "sample_id": record.sample_id,
        "sample_sha256": record.sample_sha256,
        "physical_step_sha256": record.physical_step_sha256,
        "common_objective_sha256": record.common_objective_sha256,
        "ordinal": record.ordinal,
    }
    for name, value in expected.items():
        if getattr(reference, name) != value:
            raise ValueError(f"scheduled reference {name} differs from the authenticated training payload")


def _validate_training_state(sample: V5TrainingSample, positions: torch.Tensor, contract: V5SolverContract) -> None:
    state = sample.projection_state
    if not torch.isfinite(positions).all():
        raise RuntimeError("training rollout produced non-finite positions")
    pinned_targets = sample.physical_step._owned_tensors()[-1]
    if not torch.equal(positions[state.pinned], pinned_targets):
        raise RuntimeError("training rollout changed an exact pinned target")
    deformation = compute_F(positions, state.tets, state.J)
    determinant = torch.linalg.det(deformation)
    singular = torch.linalg.svdvals(deformation)
    if not torch.isfinite(determinant).all() or not torch.isfinite(singular).all():
        raise RuntimeError("training rollout produced invalid deformation diagnostics")
    if (determinant <= contract.safeguards.minimum_determinant).any():
        raise RuntimeError("training rollout violates the checkpoint determinant bound")
    if (singular <= contract.safeguards.minimum_singular_value).any():
        raise RuntimeError("training rollout violates the checkpoint singular-value bound")


def _validate_sample_against_solver_contract(sample: V5TrainingSample, contract: V5SolverContract) -> None:
    state = sample.projection_state
    if str(state.rest_q.dtype) != contract.projection.execution_dtype:
        raise ValueError("training projection execution dtype differs from the solver contract")
    if state.operator_geometry_policy != contract.projection.operator_geometry_policy:
        raise ValueError("training operator-geometry policy differs from the solver contract")
    _verify_projection_operator(state, contract.projection)


def _representation_sample_loss(
    predictor: StretchPredictor,
    sample: V5TrainingSample,
    stage: TrainingStage,
    contract: V5SolverContract,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert stage.label_config is not None
    assert stage.representation_loss_config is not None
    state = sample.projection_state
    result = _training_sample_rollout(predictor, sample, contract)
    losses: list[torch.Tensor] = []
    for iteration in result.trace:
        labels = build_principal_stretch_labels(
            compute_F(iteration.positions_before, state.tets, state.J),
            sample.producer_attested_reference_deformation_gradient,
            stage.label_config,
        )
        loss: RepresentationLoss = principal_stretch_representation_loss(
            iteration.delta_h,
            iteration.omega,
            labels.delta_H,
            labels.omega,
            stage.representation_loss_config,
            volume=state.w,
        )
        losses.append(loss.total)
    return torch.stack(losses).mean(), result.positions


def _training_sample_rollout(
    predictor: StretchPredictor,
    sample: V5TrainingSample,
    contract: V5SolverContract,
) -> IterativeSolverResult:
    config = IterativeSolverConfig(
        iterations=contract.trained_iterations,
        detach_residual_features=contract.residual.detach_features,
        minimum_determinant=contract.safeguards.minimum_determinant,
        minimum_singular_value=contract.safeguards.minimum_singular_value,
        objective_policy="record",
        residual_policy="record",
        objective_increase_tolerance=0.0,
        normalized_residual_increase_tolerance=0.0,
        initializer_policy="persistence",
        return_projection_diagnostics=False,
        head_mode="learned",
    )
    result = solve_iterative_principal_stretch(
        predictor=predictor,
        projection_state=sample.projection_state,
        objective=sample.common_objective,
        physical_step=sample.physical_step,
        expected_physical_step_sha256=sample.sample_record.physical_step_sha256,
        config=config,
        constraint=IdentityConstraintHook(),
    )
    _validate_training_state(sample, result.positions, contract)
    return result


def compute_v5_training_batch_loss(
    *,
    bank: SharedTopologyPredictorBank,
    batch: SamplingBatch,
    payloads: dict[tuple[str, str], V5TrainingSample],
    stage: TrainingStage,
    solver_contract: V5SolverContract,
) -> V5BatchLoss:
    """Execute one scheduled batch with exact registered loss reductions."""
    if type(batch) is not SamplingBatch:
        raise TypeError("batch must be a canonical SamplingBatch")
    if type(stage) is not TrainingStage:
        raise TypeError("stage must be a canonical TrainingStage")
    if type(payloads) is not dict:
        raise TypeError("payloads must be an exact dict without callback-based lookup behavior")
    samples: list[V5TrainingSample] = []
    predictors: list[StretchPredictor] = []
    for reference in batch.samples:
        key = (reference.trajectory_id, reference.sample_id)
        sample = payloads.get(key)
        if type(sample) is not V5TrainingSample:
            raise ValueError(f"scheduled training payload {key!r} is missing")
        sample.validate_immutable()
        _validate_sample_against_solver_contract(sample, solver_contract)
        _reference_matches(reference, sample)
        if sample.sample_record.static_layout_sha256 != batch.static_layout_sha256:
            raise ValueError("training batch payload differs from the static-layout schedule")
        samples.append(sample)
        predictors.append(bank.ensure(sample))
    if len({id(predictor) for predictor in predictors}) != 1:
        raise ValueError("one static-layout batch must route through one topology predictor context")

    if stage.name == "representation":
        results = tuple(
            _representation_sample_loss(predictors[index], sample, stage, solver_contract)
            for index, sample in enumerate(samples)
        )
        total = torch.stack(tuple(result[0] for result in results)).mean()
        return V5BatchLoss(total, total, None, None, tuple(result[1] for result in results))

    if stage.name != "physics":
        raise ValueError("training stage is not registered")
    assert stage.compatible_state_loss_config is not None
    assert stage.potential_excess_loss_config is not None
    predicted = tuple(
        _training_sample_rollout(predictors[index], sample, solver_contract).positions
        for index, sample in enumerate(samples)
    )
    compatible_results: list[CompatibleStateLoss] = []
    for sample, positions in zip(samples, predicted, strict=True):
        state = sample.projection_state
        compatible_results.append(
            compatible_state_loss(
                positions,
                sample.producer_attested_reference_positions,
                tets=state.tets,
                J=state.J,
                volume=state.w,
                mass=sample.common_objective.mass,
                pinned=state.pinned,
                config=stage.compatible_state_loss_config,
            )
        )
    compatible_total = torch.stack(tuple(result.total for result in compatible_results)).mean()
    potential: PotentialExcessLoss = common_potential_excess_loss_batch(
        tuple(sample.common_objective for sample in samples),
        torch.stack(predicted),
        torch.stack(tuple(sample.producer_attested_reference_positions for sample in samples)),
        torch.stack(tuple(sample.physical_step._owned_tensors()[0] for sample in samples)),
        stage.potential_excess_loss_config,
    )
    total = compatible_total + potential.total
    if not torch.isfinite(total):
        raise RuntimeError("registered physics-stage scalar is non-finite")
    return V5BatchLoss(total, None, compatible_total, potential.total, predicted)


def _canonical_contract(contract: V5SolverContract) -> V5SolverContract:
    if type(contract) is not V5SolverContract:
        raise TypeError("solver_contract must be a canonical V5SolverContract")
    return V5SolverContract.from_dict(contract.as_dict())


def _canonical_schedule(schedule: SamplingSchedule, contract: V5SolverContract) -> SamplingSchedule:
    if type(schedule) is not SamplingSchedule:
        raise TypeError("sampling_schedule must be a canonical SamplingSchedule")
    replay = build_sampling_schedule(
        schedule.manifest,
        role=schedule.role,
        steps=schedule.steps,
        batch_size=schedule.batch_size,
        seed=schedule.seed,
    )
    if replay.as_dict() != schedule.as_dict():
        raise ValueError("sampling schedule differs from deterministic canonical replay")
    if schedule.role is not DatasetRole.TRAIN:
        raise ValueError("v5 training accepts only the frozen train schedule")
    if (
        schedule.manifest_sha256 != contract.training_dataset_sha256
        or schedule.schedule_sha256 != contract.sampling_schedule_sha256
        or schedule.steps != contract.sampling_steps
        or schedule.batch_size != contract.sampling_batch_size
        or schedule.seed != contract.sampling_seed
    ):
        raise ValueError("sampling schedule differs from the solver contract")
    return schedule


def _optimizer_options(contract: OptimizerContract, learned_dtype: torch.dtype | None = None) -> dict[str, object]:
    if contract.kind != "AdamW":
        raise ValueError("v5 trainer supports only the registered AdamW optimizer")
    options = contract.hyperparameters
    if set(options) != _ADAMW_KEYS:
        raise ValueError(f"AdamW hyperparameters must contain exactly {tuple(sorted(_ADAMW_KEYS))}")
    for name in ("learning_rate", "epsilon", "gradient_clip_norm"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"AdamW {name} must be finite and positive")
    weight_decay = options["weight_decay"]
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0.0
    ):
        raise ValueError("AdamW weight_decay must be finite and non-negative")
    for name in ("beta1", "beta2"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
            raise ValueError(f"AdamW {name} must be in [0, 1)")
    for name in ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
        if options[name] is not False:
            raise ValueError(f"AdamW {name} must be exactly false in the foundation trainer")
    if learned_dtype is not None:
        for name in ("learning_rate", "epsilon", "gradient_clip_norm"):
            materialized = torch.tensor(options[name], dtype=learned_dtype)
            if not torch.isfinite(materialized) or not (materialized > 0.0):
                raise ValueError(f"AdamW {name} must remain finite and positive in learned parameter dtype")
        materialized_weight_decay = torch.tensor(options["weight_decay"], dtype=learned_dtype)
        if not torch.isfinite(materialized_weight_decay) or (
            options["weight_decay"] > 0.0 and not (materialized_weight_decay > 0.0)
        ):
            raise ValueError("AdamW weight_decay must remain finite and preserve positivity in learned parameter dtype")
        for name in ("beta1", "beta2"):
            materialized = torch.tensor(options[name], dtype=learned_dtype)
            complement = torch.tensor(1.0, dtype=learned_dtype) - materialized
            if (
                not torch.isfinite(materialized)
                or materialized < 0.0
                or (options[name] > 0.0 and not (materialized > 0.0))
                or not (complement > 0.0)
            ):
                raise ValueError(f"AdamW {name} and its complement must be representable in learned parameter dtype")
    return options


def _make_optimizer(bank: SharedTopologyPredictorBank, contract: OptimizerContract) -> torch.optim.AdamW:
    options = _optimizer_options(contract, bank.learned_dtype)
    return torch.optim.AdamW(
        bank.shared_parameters(),
        lr=float(options["learning_rate"]),
        betas=(float(options["beta1"]), float(options["beta2"])),
        eps=float(options["epsilon"]),
        weight_decay=float(options["weight_decay"]),
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _gradient_family(name: str) -> str:
    if name.startswith("v5_context_encoder."):
        return "context_encoder"
    if name.startswith("output_head."):
        return "output_head"
    if name.startswith("rotation_head."):
        return "rotation_head"
    return "backbone"


def _gradient_norms(bank: SharedTopologyPredictorBank) -> dict[str, float]:
    totals = {"backbone": 0.0, "context_encoder": 0.0, "output_head": 0.0, "rotation_head": 0.0}
    for name, parameter in bank.master.model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"learned gradient {name!r} is non-finite")
        totals[_gradient_family(name)] += float(parameter.grad.detach().double().square().sum().item())
    return {name: math.sqrt(value) for name, value in totals.items()}


@dataclasses.dataclass(frozen=True)
class V5TrainingUpdate:
    """Auditable record for one exact scheduled optimizer update."""

    update_index: int
    stage: str
    topology_sha256: str
    samples: tuple[tuple[str, str], ...]
    loss: float
    gradient_norms: Mapping[str, float]


@dataclasses.dataclass
class V5TrainingResult:
    """Completed foundation run, shared model contexts, and verified checkpoint."""

    bank: SharedTopologyPredictorBank
    updates: tuple[V5TrainingUpdate, ...]
    access_ledger: DataAccessLedger
    checkpoint: dict[str, object]
    verified_checkpoint: VerifiedV5Checkpoint
    trainer_execution_contract_sha256: str
    training_run_record: Mapping[str, object]
    training_run_sha256: str
    completed_updates: int
    next_batch_index: int
    optimizer_parameter_names: tuple[str, ...]


def _stage_for_update(stages: Sequence[TrainingStage], update_index: int) -> TrainingStage:
    matches = tuple(stage for stage in stages if stage.start_update <= update_index < stage.end_update)
    if len(matches) != 1:
        raise RuntimeError("ordered training stage plan does not uniquely cover the schedule update")
    return matches[0]


def _validate_manifest_payload_membership(
    schedule: SamplingSchedule,
    payloads: dict[tuple[str, str], V5TrainingSample],
) -> None:
    if type(payloads) is not dict:
        raise TypeError("payloads must be an exact dict without callback-based lookup behavior")
    expected_keys = {
        (reference.trajectory_id, reference.sample_id) for batch in schedule.batches for reference in batch.samples
    }
    if set(payloads) != expected_keys:
        raise ValueError("training payload map must contain exactly the samples referenced by the schedule")
    for key, payload in payloads.items():
        if type(payload) is not V5TrainingSample or payload.key != key:
            raise ValueError("training payload map key differs from its sealed sample identity")
        if schedule.manifest.role_for_trajectory(payload.trajectory_id) is not DatasetRole.TRAIN:
            raise ValueError("non-train payloads are forbidden in the executable trainer")
        trajectory = schedule.manifest.trajectory(payload.trajectory_id)
        records = tuple(sample for sample in trajectory.samples if sample.sample_id == payload.sample_record.sample_id)
        if len(records) != 1 or records[0].as_dict() != payload.sample_record.as_dict():
            raise ValueError("training payload sample record differs from the frozen SplitManifest")


def _validated_ledger(ledger: DataAccessLedger, schedule: SamplingSchedule) -> DataAccessLedger:
    if type(ledger) is not DataAccessLedger:
        raise TypeError("access_ledger must be a canonical DataAccessLedger")
    rebuilt = DataAccessLedger(ledger.manifest, ledger.accesses)
    if rebuilt.as_dict() != ledger.as_dict():
        raise ValueError("data-access ledger changed after authentication")
    if ledger.manifest.manifest_sha256 != schedule.manifest_sha256:
        raise ValueError("data-access ledger and training schedule bind different manifests")
    if ledger.confirmation_payload_released:
        raise ValueError("training cannot run after confirmation payload release on this ledger branch")
    return ledger


def _rng_snapshot(
    model_initialization_seed: int,
    sampling_seed: int,
    bank: SharedTopologyPredictorBank,
    training_run_record: Mapping[str, object],
    training_run_sha256: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "torch_cpu": torch.get_rng_state(),
        "model_initialization_torch_seed": model_initialization_seed,
        "numpy_pcg64_sampling_schedule_seed": sampling_seed,
        "trainer_execution_contract": trainer_execution_contract(),
        "trainer_execution_contract_sha256": TRAINER_EXECUTION_CONTRACT_SHA256,
        "training_run_record": dict(training_run_record),
        "training_run_sha256": training_run_sha256,
        "parameter_order": list(bank.parameter_names),
        "resume_capability": False,
    }
    if torch.cuda.is_available():
        result["torch_cuda_all"] = tuple(torch.cuda.get_rng_state_all())
    return result


def _optimizer_snapshot(optimizer: torch.optim.AdamW, bank: SharedTopologyPredictorBank) -> dict[str, object]:
    return {
        "optimizer_state_dict": optimizer.state_dict(),
        "parameter_order": list(bank.parameter_names),
        "parameter_count": len(bank.parameter_names),
        "resume_capability": False,
        "semantics": "integrity-only-not-a-resume-proof",
    }


def _training_run_record(
    *,
    contract: V5SolverContract,
    schedule: SamplingSchedule,
    ledger: DataAccessLedger,
    updates: Sequence[V5TrainingUpdate],
    parameter_names: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": "pss-v5-training-run-integrity-record-v1",
        "claim_scope": "integrity-only-not-exact-resume-or-model-selection-evidence",
        "solver_contract_sha256": contract.solver_contract_sha256,
        "sampling_schedule_sha256": schedule.schedule_sha256,
        "trainer_execution_contract_sha256": TRAINER_EXECUTION_CONTRACT_SHA256,
        "final_access_ledger_sha256": ledger.ledger_sha256,
        "final_access_ledger": ledger.as_dict(),
        "updates": [
            {
                "update_index": update.update_index,
                "stage": update.stage,
                "topology_sha256": update.topology_sha256,
                "samples": [list(key) for key in update.samples],
                "loss": update.loss,
                "gradient_norms": dict(sorted(update.gradient_norms.items())),
            }
            for update in updates
        ],
        "completed_updates": len(updates),
        "next_batch_index": len(updates),
        "parameter_order": list(parameter_names),
        "resume_capability": False,
    }


def train_pr_history_v5(
    *,
    solver_contract: V5SolverContract,
    sampling_schedule: SamplingSchedule,
    access_ledger: DataAccessLedger,
    payloads: dict[tuple[str, str], V5TrainingSample],
    seed: int = 0,
    checkpoint_path: str | pathlib.Path | None = None,
) -> V5TrainingResult:
    """Train one architecture-v5 model over the exact authenticated schedule.

    The function always starts a new root run.  It intentionally exposes no
    resume argument because the current checkpoint contract authenticates an
    optimizer/RNG snapshot but does not prove exact next-update replay.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    contract = _canonical_contract(solver_contract)
    schedule = _canonical_schedule(sampling_schedule, contract)
    _validate_manifest_payload_membership(schedule, payloads)
    ledger = _validated_ledger(access_ledger, schedule)
    if contract.projection.backend != "dense":
        raise ValueError("executable v5 training requires the dense differentiable projection contract")
    learned_dtype = _LEARNED_DTYPES[contract.learned_parameter_dtype]
    graph_config = GraphTransformerConfig(**contract.graph_config)
    optimizer_options = _optimizer_options(contract.optimizer, learned_dtype)

    torch.manual_seed(seed)
    bank = SharedTopologyPredictorBank(graph_config, learned_dtype)
    optimizer: torch.optim.AdamW | None = None
    accessed_trajectories: set[str] = set()
    updates: list[V5TrainingUpdate] = []
    for update_index, batch in enumerate(schedule.batches):
        for reference in batch.samples:
            if reference.trajectory_id not in accessed_trajectories:
                ledger = ledger.record_access(
                    reference.trajectory_id,
                    purpose=DataAccessPurpose.TRAINING,
                    scope=DataAccessScope.PAYLOAD,
                    payload_names=_OPENED_PAYLOAD_NAMES,
                )
                accessed_trajectories.add(reference.trajectory_id)
            payloads[(reference.trajectory_id, reference.sample_id)].validate_immutable()
        first_reference = batch.samples[0]
        bank.ensure(payloads[(first_reference.trajectory_id, first_reference.sample_id)])
        if optimizer is None:
            optimizer = _make_optimizer(bank, contract.optimizer)
        stage = _stage_for_update(contract.stages, update_index)
        optimizer.zero_grad(set_to_none=True)
        batch_loss = compute_v5_training_batch_loss(
            bank=bank,
            batch=batch,
            payloads=payloads,
            stage=stage,
            solver_contract=contract,
        )
        if not torch.isfinite(batch_loss.total):
            raise RuntimeError("training batch loss is non-finite")
        batch_loss.total.backward()
        gradient_norms = _gradient_norms(bank)
        preclip_norm = torch.nn.utils.clip_grad_norm_(
            bank.shared_parameters(),
            float(optimizer_options["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        if not torch.isfinite(preclip_norm):
            raise RuntimeError("aggregate learned gradient norm is non-finite")
        optimizer.step()
        for name, parameter in bank.master.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"AdamW produced non-finite learned parameter {name!r}")
        updates.append(
            V5TrainingUpdate(
                update_index=update_index,
                stage=stage.name,
                topology_sha256=batch.topology_sha256,
                samples=tuple((reference.trajectory_id, reference.sample_id) for reference in batch.samples),
                loss=float(batch_loss.total.detach().cpu().item()),
                gradient_norms=gradient_norms,
            )
        )
    if optimizer is None:
        raise RuntimeError("canonical sampling schedule unexpectedly contained no updates")
    bank._validate_shared_parameter_identity()
    training_run_record = _training_run_record(
        contract=contract,
        schedule=schedule,
        ledger=ledger,
        updates=updates,
        parameter_names=bank.parameter_names,
    )
    training_run_sha256 = canonical_json_sha256(training_run_record)

    checkpoint = build_v5_checkpoint(
        bank.master.model.state_dict(),
        solver_contract=contract,
        optimizer_state=_optimizer_snapshot(optimizer, bank),
        rng_state=_rng_snapshot(seed, schedule.seed, bank, training_run_record, training_run_sha256),
        batch_stream=schedule,
        completed_updates=len(updates),
        parent_lineage=ParentLineage.root(),
    )
    verified = verify_v5_checkpoint(checkpoint)
    if verified.completed_updates != len(updates):
        raise RuntimeError("verified checkpoint completed-update count differs from schedule consumption")
    if checkpoint_path is not None:
        save_verified_v5_checkpoint(checkpoint, checkpoint_path)
    return V5TrainingResult(
        bank=bank,
        updates=tuple(updates),
        access_ledger=ledger,
        checkpoint=checkpoint,
        verified_checkpoint=verified,
        trainer_execution_contract_sha256=TRAINER_EXECUTION_CONTRACT_SHA256,
        training_run_record=training_run_record,
        training_run_sha256=training_run_sha256,
        completed_updates=len(updates),
        next_batch_index=len(updates),
        optimizer_parameter_names=bank.parameter_names,
    )


def save_verified_v5_checkpoint(checkpoint: Mapping[str, object], path: str | pathlib.Path) -> VerifiedV5Checkpoint:
    """Verify, atomically save, reload, and reverify one schema-v5 checkpoint."""
    verified = verify_v5_checkpoint(checkpoint)
    destination = pathlib.Path(path)
    if not destination.parent.is_dir():
        raise ValueError("checkpoint parent directory does not exist")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as stream:
            temporary_name = stream.name
        torch.save(dict(checkpoint), temporary_name)
        loaded = torch.load(temporary_name, map_location="cpu", weights_only=False)
        reverified = verify_v5_checkpoint(loaded)
        if reverified != verified:
            raise RuntimeError("saved checkpoint verification differs from the in-memory artifact")
        os.replace(temporary_name, destination)
        temporary_name = None
        return reverified
    finally:
        if temporary_name is not None:
            pathlib.Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "TRAINER_EXECUTION_CONTRACT_SHA256",
    "SharedTopologyPredictorBank",
    "V5BatchLoss",
    "V5TrainingResult",
    "V5TrainingSample",
    "V5TrainingUpdate",
    "build_v5_adamw_optimizer_contract",
    "canonical_training_tensor_sha256",
    "compute_v5_training_batch_loss",
    "save_verified_v5_checkpoint",
    "train_pr_history_v5",
    "trainer_execution_contract",
]
