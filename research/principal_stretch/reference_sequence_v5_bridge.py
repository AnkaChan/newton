# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Lazy authenticated bridge from reference sequences to v5 training samples.

The durable loader in :mod:`reference_sequence_dataset` owns file and
producer-evidence authentication.  This module is the next trust boundary: it
converts one authenticated transition at a time into the exact runtime
contexts consumed by the v5 trainer.  Static dense projection state is cached
once per asset, while dynamic transition tensors are deliberately not cached.

Producer identities and v5 runtime identities remain separate.  In
particular, the producer operator covers a broader static inventory and its
material identity covers the stored SolverVBD arrays; neither is substituted
for the v5 source-pose operator or runtime mass/material identity.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from .iterative_solver import (
    PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
    PhysicalStepContext,
    SolverVBDStagedFloat32Evidence,
    validate_physical_objective_integration,
)
from .reference_sequence_dataset import (
    ReferenceSequenceDataset,
    ReferenceTransition,
    ReferenceTransitionKey,
)
from .torch_solver import (
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
    SolverState,
    build_solver,
    compute_F,
    projection_state_sha256,
    validate_authenticated_operator_geometry,
)
from .train_pr_history_v5 import V5TrainingSample, canonical_training_tensor_sha256
from .v5_checkpoint import canonical_json_sha256
from .v5_dataset import DatasetRole, NumericContentIdentity, TrajectorySampleRecord
from .v5_objective import CommonObjectiveContext
from .v5_pr_history_loader import (
    canonical_runtime_material_sha256,
    canonical_runtime_pin_signature_sha256,
)

_SOURCE_TRANSITION_CONTRACT = "pss-reference-sequence-v5-source-transition-v1"
_NUMERIC_IDENTIFIER_PREFIX = "reference-sequence-v5"


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


@dataclasses.dataclass(frozen=True, order=True)
class ReferenceV5AssetIdentities:
    """Separate producer and v5 identities for one cached asset context."""

    asset_id: str
    asset_source_sha256: str
    static_npz_sha256: str
    producer_topology_sha256: str
    producer_operator_sha256: str
    producer_material_sha256: str
    v5_topology_sha256: str
    v5_operator_geometry_sha256: str
    v5_material_sha256: str
    v5_pin_signature_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.asset_id, "asset_id")
        for name in (
            "asset_source_sha256",
            "static_npz_sha256",
            "producer_topology_sha256",
            "producer_operator_sha256",
            "producer_material_sha256",
            "v5_topology_sha256",
            "v5_operator_geometry_sha256",
            "v5_material_sha256",
            "v5_pin_signature_sha256",
        ):
            _sha256(getattr(self, name), name)


@dataclasses.dataclass(frozen=True)
class ReferenceV5MaterializedSample:
    """One lazy v5 payload plus the producer identities from which it came."""

    transition_key: ReferenceTransitionKey
    identities: ReferenceV5AssetIdentities
    source_transition_sha256: str
    training_sample: V5TrainingSample

    def __post_init__(self) -> None:
        if type(self.transition_key) is not ReferenceTransitionKey:
            raise TypeError("transition_key must be a canonical ReferenceTransitionKey")
        if type(self.identities) is not ReferenceV5AssetIdentities:
            raise TypeError("identities must be canonical ReferenceV5AssetIdentities")
        _sha256(self.source_transition_sha256, "source_transition_sha256")
        if type(self.training_sample) is not V5TrainingSample:
            raise TypeError("training_sample must be a canonical V5TrainingSample")
        self.training_sample.validate_immutable()
        expected_trajectory = _trajectory_id(self.transition_key)
        expected_sample = _sample_id(self.transition_key)
        if (
            self.training_sample.trajectory_id != expected_trajectory
            or self.training_sample.sample_record.sample_id != expected_sample
            or self.training_sample.sample_record.ordinal != self.transition_key.step_id
        ):
            raise ValueError("training-sample identity differs from its reference transition")
        evidence = self.training_sample.physical_step.source_evidence
        if (
            type(evidence) is not SolverVBDStagedFloat32Evidence
            or evidence.source_transition_sha256 != self.source_transition_sha256
        ):
            raise ValueError("training source evidence differs from the reference transition identity")
        record = self.training_sample.sample_record
        if (
            self.identities.asset_id != self.transition_key.asset_id
            or self.identities.v5_topology_sha256 != record.topology_sha256
            or self.identities.v5_operator_geometry_sha256 != record.operator_geometry_sha256
            or self.identities.v5_material_sha256 != record.material_sha256
            or self.identities.v5_pin_signature_sha256 != record.pin_signature_sha256
        ):
            raise ValueError("asset identities differ from the sealed v5 sample")

    @property
    def key(self) -> tuple[str, str]:
        """Return the exact lookup key consumed by the v5 trainer."""
        return self.training_sample.key


@dataclasses.dataclass(frozen=True)
class _AssetContext:
    identities: ReferenceV5AssetIdentities
    projection_state: SolverState


def _trajectory_id(key: ReferenceTransitionKey) -> str:
    return f"reference-sequence:{key.asset_id}:{key.sequence_id}"


def _sample_id(key: ReferenceTransitionKey) -> str:
    return f"step-{key.step_id:08d}"


def _floating(value: np.ndarray, device: torch.device) -> torch.Tensor:
    source = np.array(value, dtype=np.float64, order="C", copy=True)
    return torch.as_tensor(source, dtype=torch.float64, device=device)


def _source_float32(value: np.ndarray, device: torch.device, name: str) -> torch.Tensor:
    raw = np.ascontiguousarray(np.asarray(value))
    source = np.array(raw, dtype=np.float32, order="C", copy=True)
    if raw.dtype == np.float64:
        roundtrip = source.astype(np.float64)
        if raw.shape != roundtrip.shape or raw.tobytes(order="C") != roundtrip.tobytes(order="C"):
            raise ValueError(f"{name} must be an exact float32 image")
    elif raw.dtype != np.float32:
        raise ValueError(f"{name} must be float32 or its lossless float64 promotion")
    return torch.as_tensor(source, dtype=torch.float32, device=device)


def _source_transition_sha256(dataset: ReferenceSequenceDataset, transition: ReferenceTransition) -> str:
    """Bind one step to the index that transitively seals accepted evidence.

    ``ReferenceSequenceDataset.index_sha256`` seals each producer-manifest
    digest.  Each verified producer manifest in turn seals the evidence JSON,
    whose selected step and acceptance gate were checked by the durable
    loader.  Including the index digest here therefore binds the staged
    physical evidence to that accepted-step record without reaching through
    the loader's private producer representation.
    """
    key = transition.key
    return canonical_json_sha256(
        {
            "contract": _SOURCE_TRANSITION_CONTRACT,
            "reference_sequence_index_sha256": dataset.index_sha256,
            "asset_id": key.asset_id,
            "asset_source_sha256": transition.asset_source_sha256,
            "sequence_id": key.sequence_id,
            "step_id": key.step_id,
            "sequence_npz_sha256": transition.sequence_npz_sha256,
            "protocol_sha256": transition.protocol_sha256,
            "producer_topology_sha256": transition.topology_sha256,
            "producer_operator_sha256": transition.operator_sha256,
            "producer_material_sha256": transition.material_sha256,
            "accepted_reference_state_sha256": transition.reference_state_float64_sha256,
        }
    )


def _numeric_identity(
    dataset: ReferenceSequenceDataset,
    source_transition_sha256: str,
    name: str,
    tensor: torch.Tensor,
) -> NumericContentIdentity:
    return NumericContentIdentity(
        identifier=f"{_NUMERIC_IDENTIFIER_PREFIX}:{dataset.index_sha256}:{source_transition_sha256}:{name}",
        sha256=canonical_training_tensor_sha256(tensor),
    )


class ReferenceSequenceV5Bridge:
    """Lazily materialize authenticated reference transitions on one device.

    A bridge instance owns no transition cache.  Its only retained runtime
    payload is one dense, source-pose-authenticated projection state per exact
    asset identity.  The existing :class:`SharedTopologyPredictorBank` then
    provides the corresponding one-static-graph-per-topology cache while
    sharing learned parameters across assets.
    """

    def __init__(self, dataset: ReferenceSequenceDataset, *, device: str | torch.device = "cpu") -> None:
        if type(dataset) is not ReferenceSequenceDataset:
            raise TypeError("dataset must be a canonical ReferenceSequenceDataset")
        self.dataset = dataset
        self.device = torch.device(device)
        self._asset_contexts: dict[str, _AssetContext] = {}

    @property
    def cached_asset_count(self) -> int:
        """Return the number of constructed static projection contexts."""
        return len(self._asset_contexts)

    @property
    def cached_asset_identities(self) -> tuple[ReferenceV5AssetIdentities, ...]:
        """Return cached asset identities in deterministic asset-id order."""
        return tuple(self._asset_contexts[asset_id].identities for asset_id in sorted(self._asset_contexts))

    def sample_keys(
        self,
        role: DatasetRole | str,
        *,
        count: int,
        seed: int,
    ) -> tuple[ReferenceTransitionKey, ...]:
        """Replay the durable dataset's exact stateless transition sampler."""
        return self.dataset.sample_keys(role, count=count, seed=seed)

    def iter_materialized(
        self,
        keys: Iterable[ReferenceTransitionKey],
    ) -> Iterator[ReferenceV5MaterializedSample]:
        """Yield samples one at a time without retaining dynamic payloads."""
        for key in keys:
            yield self.materialize(key)

    def materialize(self, key: ReferenceTransitionKey) -> ReferenceV5MaterializedSample:
        """Authenticate and construct one trainer-ready v5 transition."""
        if type(key) is not ReferenceTransitionKey:
            raise TypeError("key must be a canonical ReferenceTransitionKey")
        transition = self.dataset.transition(key)
        context = self._asset_context(transition)
        source_transition_sha256 = _source_transition_sha256(self.dataset, transition)
        training_sample = self._training_sample(transition, context, source_transition_sha256)
        return ReferenceV5MaterializedSample(
            transition_key=key,
            identities=context.identities,
            source_transition_sha256=source_transition_sha256,
            training_sample=training_sample,
        )

    def _asset_context(self, transition: ReferenceTransition) -> _AssetContext:
        static = transition.static
        if (
            transition.topology_sha256 != static.topology_sha256
            or transition.operator_sha256 != static.operator_sha256
            or transition.material_sha256 != static.material_sha256
        ):
            raise ValueError("transition producer identities differ from its authenticated static shard")
        cached = self._asset_contexts.get(transition.key.asset_id)
        if cached is not None:
            identity = cached.identities
            if (
                identity.asset_source_sha256 != transition.asset_source_sha256
                or identity.static_npz_sha256 != static.static_npz_sha256
                or identity.producer_topology_sha256 != transition.topology_sha256
                or identity.producer_operator_sha256 != transition.operator_sha256
                or identity.producer_material_sha256 != transition.material_sha256
                or identity.v5_topology_sha256 != static.v5_topology_sha256
                or identity.v5_operator_geometry_sha256 != static.v5_operator_geometry_sha256
            ):
                raise ValueError("one asset_id resolved to conflicting static identities")
            return cached

        state = build_solver(
            np.array(static.v5_source_rest_q, dtype=np.float32, order="C", copy=True),
            np.array(static.v5_source_tet_indices, dtype=np.int64, order="C", copy=True),
            np.array(static.v5_source_tet_poses, dtype=np.float32, order="C", copy=True),
            np.empty((0,), dtype=np.int64),
            self.device,
            dtype=torch.float64,
            tikhonov=0.0,
            projection_backend="dense",
            operator_geometry_policy=OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
            translation_gauge_policy=TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
            vertex_masses=np.array(static.mass, dtype=np.float64, order="C", copy=True),
        )
        operator_sha256 = validate_authenticated_operator_geometry(state)
        if state.static_mesh_sha256 != static.v5_topology_sha256:
            raise ValueError("cached v5 topology differs from the authenticated static shard")
        if operator_sha256 != static.v5_operator_geometry_sha256:
            raise ValueError("cached v5 source-pose operator differs from the authenticated static shard")
        if state.projection_state_sha256 != projection_state_sha256(state):
            raise ValueError("cached projection state failed self-authentication")

        mass = _floating(static.mass, self.device)
        mu = _floating(static.tet_materials[:, 0], self.device)
        lam = _floating(static.tet_materials[:, 1], self.device)
        pinned = torch.empty((0,), dtype=torch.int64, device=self.device)
        material_sha256 = canonical_runtime_material_sha256(mass, mu, lam)
        pin_sha256 = canonical_runtime_pin_signature_sha256(pinned, state.tets, state.n_verts)
        identities = ReferenceV5AssetIdentities(
            asset_id=transition.key.asset_id,
            asset_source_sha256=transition.asset_source_sha256,
            static_npz_sha256=static.static_npz_sha256,
            producer_topology_sha256=transition.topology_sha256,
            producer_operator_sha256=transition.operator_sha256,
            producer_material_sha256=transition.material_sha256,
            v5_topology_sha256=state.static_mesh_sha256,
            v5_operator_geometry_sha256=operator_sha256,
            v5_material_sha256=material_sha256,
            v5_pin_signature_sha256=pin_sha256,
        )
        context = _AssetContext(identities=identities, projection_state=state)
        self._asset_contexts[transition.key.asset_id] = context
        return context

    def _training_sample(
        self,
        transition: ReferenceTransition,
        context: _AssetContext,
        source_transition_sha256: str,
    ) -> V5TrainingSample:
        static = transition.static
        state = context.projection_state
        dtype = torch.float64
        mass = _floating(static.mass, self.device)
        mu = _floating(static.tet_materials[:, 0], self.device)
        lam = _floating(static.tet_materials[:, 1], self.device)
        pinned = torch.empty((0,), dtype=torch.int64, device=self.device)
        inertial_target = _floating(transition.inertial_target, self.device)
        common_objective = CommonObjectiveContext(
            tets=state.tets,
            J=state.J,
            volume=state.w,
            mass=mass,
            mu=mu,
            lam=lam,
            inertial_target=inertial_target,
            pinned=pinned,
            dt=float(transition.execution_dt_seconds),
        )

        x_current = _floating(transition.x_current, self.device)
        x_previous = _floating(transition.x_previous, self.device)
        force = _floating(transition.external_force, self.device)
        gravity = _floating(transition.gravity, self.device)
        pin = torch.zeros((state.n_tets,), dtype=dtype, device=self.device)
        pinned_targets = torch.empty((0, 3), dtype=dtype, device=self.device)
        source_evidence = SolverVBDStagedFloat32Evidence(
            source_transition_sha256=source_transition_sha256,
            dt_seconds=float(transition.execution_dt_seconds),
            pre_event_positions=_source_float32(transition.x_current, self.device, "pre-event positions"),
            velocity=_source_float32(transition.velocity, self.device, "velocity"),
            mass=_source_float32(static.mass, self.device, "mass"),
            inverse_mass=_source_float32(static.particle_inv_mass, self.device, "inverse mass"),
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
        validate_physical_objective_integration(state, common_objective, physical_step)

        input_state = x_current.clone()
        reference_state = _floating(transition.reference_positions, self.device)
        observed_f = compute_F(x_current, state.tets, state.J)
        input_f = compute_F(input_state, state.tets, state.J)
        reference_f = compute_F(reference_state, state.tets, state.J)
        tensors = {
            "observed_f": observed_f,
            "input_f": input_f,
            "reference_f": reference_f,
            "observed_state": x_current,
            "input_state": input_state,
            "reference_state": reference_state,
        }
        numeric = {
            name: _numeric_identity(self.dataset, source_transition_sha256, name, tensor)
            for name, tensor in tensors.items()
        }
        record = TrajectorySampleRecord(
            sample_id=_sample_id(transition.key),
            ordinal=transition.key.step_id,
            topology_sha256=context.identities.v5_topology_sha256,
            operator_geometry_sha256=context.identities.v5_operator_geometry_sha256,
            material_sha256=context.identities.v5_material_sha256,
            pin_signature_sha256=context.identities.v5_pin_signature_sha256,
            dt_seconds=float(transition.execution_dt_seconds),
            physical_step_sha256=physical_step.physical_step_sha256,
            physical_integration_policy=physical_step.integration_policy,
            source_integration_evidence_sha256=source_evidence.evidence_sha256,
            common_objective_sha256=common_objective.common_objective_sha256,
            observed_f=numeric["observed_f"],
            input_f=numeric["input_f"],
            reference_f=numeric["reference_f"],
            observed_state=numeric["observed_state"],
            input_state=numeric["input_state"],
            reference_state=numeric["reference_state"],
        )
        return V5TrainingSample(
            trajectory_id=_trajectory_id(transition.key),
            sample_record=record,
            physical_step=physical_step,
            common_objective=common_objective,
            projection_state=state,
            producer_attested_reference_positions=reference_state,
            producer_attested_reference_deformation_gradient=reference_f,
        )


__all__ = [
    "ReferenceSequenceV5Bridge",
    "ReferenceV5AssetIdentities",
    "ReferenceV5MaterializedSample",
]
