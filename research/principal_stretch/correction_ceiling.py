# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dense residual-correction ceiling for principal-stretch research.

The bounded prefixes here answer one architectural question: after an
initializer such as one VBD sweep, can one or two unregularized, globally
safeguarded Newton corrections reach a target VBD budget on the same
implicit-Euler objective?  This is a quality ceiling, not a fast solver.  It
uses a dense float64 Hessian and its timings cannot support performance
claims.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import numbers
import types
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .newton_baseline import (
    NewtonConfig,
    NewtonProblem,
    NewtonResidualPolishConfig,
    solve_newton_residual_polish,
)
from .pr_scene_history import (
    AppliedAtomicState,
    AtomicCoordinate,
    CommittedState,
    FailedReference,
    HistoryCheckpoint,
    HistoryTransition,
    PRHistoryChain,
    PRHistoryManifest,
    PRSceneHistory,
    TransitionTiming,
)
from .solver_benchmark import (
    CommonStateMetrics,
    TetBenchmarkScene,
    VBDRunResult,
    _array_digest,
    _canonical_digest,
    _vbd_run_digest,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
    run_vbd,
)

_CONTRACT = "pss-dense-residual-correction-prefix-v1"
_STATE_CONTRACT = "pss-correction-state-evidence-v1"
_COMPARISON_CONTRACT = "pss-correction-comparison-v1"
_LADDER_CONTRACT = "pss-correction-ladder-v1"
_TRANSITION_CONTRACT = "pss-pr-transition-correction-ceiling-v1"
_DIAGNOSTIC_TIMING_PROVENANCE = "self-reported-diagnostic-not-performance-evidence-v1"
_HISTORY_KINDS = ("stretch", "twist", "compression-50", "compression-90")
_OBJECTIVE_ROUNDOFF_FACTOR = 8.0
_REFERENCE_METHOD_POLICIES = {
    "dense-cpu-newton-float64": None,
    "dense-cpu-newton-float64-with-strict-residual-polish": (
        "residual_polish_policy",
        "strict-reference-residual-newton-three-start-v1",
    ),
    "dense-cpu-newton-float64-with-alternate-residual-verification": (
        "alternate_residual_policy",
        "alternate-start-only-residual-verification-v1",
    ),
}
_RESIDUAL_POLISH_REASONS = {
    "gradient",
    "max_iterations",
    "nonfinite",
    "nonfinite_hessian",
    "non_spd_hessian",
    "factorization",
    "non_descent",
    "residual_line_search",
}


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("evidence mapping keys must be canonical strings")
            frozen[key] = _freeze_json(item)
        return types.MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError("evidence values must use canonical JSON scalar types")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _readonly_positions(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = value
    contiguous = np.array(array, dtype=np.float64, order="C", copy=True)
    if contiguous.ndim != 2 or contiguous.shape[1] != 3:
        raise ValueError(f"{name} must have shape (V, 3)")
    if not np.isfinite(contiguous).all():
        raise ValueError(f"{name} must be finite")
    # A normal owning NumPy array can reverse ``write=False`` with
    # ``setflags(write=True)``.  Rebuild the view over immutable ``bytes`` so
    # callers cannot mutate an already authenticated state and leave its
    # cached digest stale.
    positions = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(contiguous.shape)
    return positions


def _finite_metrics(metrics: CommonStateMetrics) -> bool:
    required = (
        metrics.objective,
        metrics.inertia,
        metrics.elastic,
        metrics.gradient_norm,
        metrics.relative_residual,
        metrics.determinant_min,
        metrics.determinant_max,
        metrics.inverted_tet_fraction,
        metrics.minimum_singular_value,
        metrics.max_pin_error_m,
    )
    optional = (metrics.free_rms_error_m, metrics.mass_weighted_rms_error_m)
    if not all(type(value) is float and math.isfinite(value) for value in required) or not all(
        value is None or (type(value) is float and math.isfinite(value)) for value in optional
    ):
        return False
    digest = metrics.position_sha256
    if type(digest) is not str or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return False
    if (
        metrics.inertia < 0.0
        or metrics.gradient_norm < 0.0
        or metrics.relative_residual < 0.0
        or metrics.determinant_min > metrics.determinant_max
        or not 0.0 <= metrics.inverted_tet_fraction <= 1.0
        or metrics.minimum_singular_value < 0.0
        or metrics.max_pin_error_m < 0.0
        or any(value is not None and value < 0.0 for value in optional)
    ):
        return False
    decomposition_scale = max(1.0, abs(metrics.objective), abs(metrics.inertia), abs(metrics.elastic))
    decomposition_guard = _OBJECTIVE_ROUNDOFF_FACTOR * np.finfo(np.float64).eps * decomposition_scale
    return abs(metrics.objective - (metrics.inertia + metrics.elastic)) <= decomposition_guard


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _verify_self_hashed_record(record: Mapping[str, object], hash_field: str, label: str) -> None:
    payload = dict(_thaw_json(record))
    supplied = payload.pop(hash_field, None)
    _sha256(supplied, f"{label} {hash_field}")
    if supplied != _canonical_digest(payload):
        raise ValueError(f"{label} raw content changed after authentication")


def _validate_canonical_frozen_json(value: object, label: str) -> None:
    """Require the immutable container representation produced by PR history."""
    if type(value) is types.MappingProxyType:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{label} has a noncanonical mapping key")
            _validate_canonical_frozen_json(item, label)
        return
    if type(value) is tuple:
        for item in value:
            _validate_canonical_frozen_json(item, label)
        return
    if value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value)):
        return
    raise ValueError(f"{label} is not canonical immutable JSON")


def _validate_canonical_committed_state(state: CommittedState, label: str) -> None:
    if type(state) is not CommittedState or type(state.coordinate) is not AtomicCoordinate:
        raise ValueError(f"{label} has a noncanonical state type")
    _sha256(state.manifest_sha256, f"{label} manifest_sha256")
    _sha256(state.state_sha256, f"{label} state_sha256")
    if any(type(value) is not np.ndarray for value in (state.q, state.qd, state.particle_flags)):
        raise ValueError(f"{label} has a noncanonical state array")


def _validate_canonical_checkpoint(checkpoint: HistoryCheckpoint, label: str) -> None:
    if type(checkpoint) is not HistoryCheckpoint:
        raise ValueError(f"{label} has a noncanonical checkpoint type")
    _validate_canonical_committed_state(checkpoint.state, f"{label} state")
    _sha256(checkpoint.manifest_sha256, f"{label} manifest_sha256")
    _sha256(checkpoint.prefix_sha256, f"{label} prefix_sha256")
    _sha256(checkpoint.checkpoint_sha256, f"{label} checkpoint_sha256")
    if checkpoint.prior_transition_sha256 is not None:
        _sha256(checkpoint.prior_transition_sha256, f"{label} prior_transition_sha256")


def _validate_canonical_history_manifest(manifest: PRHistoryManifest) -> None:
    if type(manifest) is not PRHistoryManifest:
        raise ValueError("PR history chain has a noncanonical manifest type")
    if type(manifest.kind) is not str or manifest.kind not in _HISTORY_KINDS:
        raise ValueError("PR history manifest has a noncanonical kind")
    for name in ("source_path", "source_revision"):
        value = getattr(manifest, name)
        if type(value) is not str or not value:
            raise ValueError(f"PR history manifest has a noncanonical {name}")
    for name in ("base_physical_sha256", "topology_sha256", "material_sha256", "manifest_sha256"):
        _sha256(getattr(manifest, name), f"PR history manifest {name}")
    if manifest.compression_ratio is not None and type(manifest.compression_ratio) is not float:
        raise ValueError("PR history manifest has a noncanonical compression ratio")
    for name in ("total_frames", "substeps_per_frame", "schema_version"):
        if type(getattr(manifest, name)) is not int:
            raise ValueError(f"PR history manifest has a noncanonical {name}")
    if type(manifest.dt_seconds) is not float:
        raise ValueError("PR history manifest has a noncanonical time step")
    _validate_canonical_frozen_json(manifest.schedule, "PR history manifest schedule")


def _validate_canonical_transition_container(transition: HistoryTransition, label: str) -> None:
    if type(transition) is not HistoryTransition:
        raise ValueError(f"{label} has a noncanonical transition type")
    if type(transition.coordinate) is not AtomicCoordinate or type(transition.next_coordinate) is not AtomicCoordinate:
        raise ValueError(f"{label} has a noncanonical coordinate type")
    _validate_canonical_committed_state(transition.input_state, f"{label} input")
    _validate_canonical_committed_state(transition.output_state, f"{label} output")
    applied = transition.applied_state
    if type(applied) is not AppliedAtomicState or type(applied.coordinate) is not AtomicCoordinate:
        raise ValueError(f"{label} has a noncanonical applied-state type")
    if type(applied.callback_applied) is not bool or type(applied.action) is not str:
        raise ValueError(f"{label} has noncanonical applied-state scalars")
    if applied.schedule_value_name is not None and type(applied.schedule_value_name) is not str:
        raise ValueError(f"{label} has a noncanonical schedule-value name")
    if applied.schedule_value is not None and type(applied.schedule_value) is not float:
        raise ValueError(f"{label} has a noncanonical schedule value")
    if any(
        type(value) is not np.ndarray
        for value in (applied.q, applied.particle_flags, applied.pinned_indices, applied.pin_targets)
    ):
        raise ValueError(f"{label} has a noncanonical applied-state array")
    _validate_canonical_frozen_json(transition.applied_record, f"{label} applied record")
    _validate_canonical_frozen_json(transition.reference_record, f"{label} reference record")
    if type(transition.dt_seconds) is not float:
        raise ValueError(f"{label} has a noncanonical time step")
    if type(transition.inertial_target) is not np.ndarray or type(transition.reference_positions) is not np.ndarray:
        raise ValueError(f"{label} has a noncanonical transition array")
    for name in (
        "manifest_sha256",
        "input_state_sha256",
        "input_prefix_sha256",
        "scene_sha256",
        "objective_instance_sha256",
        "topology_sha256",
        "material_sha256",
        "transition_sha256",
    ):
        _sha256(getattr(transition, name), f"{label} {name}")


def _validate_canonical_chain_container(chain: PRHistoryChain) -> None:
    if type(chain) is not PRHistoryChain or "verify" in vars(chain) or "checkpoint_at" in vars(chain):
        raise ValueError("correction evidence requires the canonical PRHistoryChain implementation")
    _validate_canonical_history_manifest(chain.manifest)
    _validate_canonical_checkpoint(chain.initial_checkpoint, "chain initial checkpoint")
    _validate_canonical_checkpoint(chain.final_checkpoint, "chain final checkpoint")
    if type(chain.transitions) is not tuple or type(chain.timings) is not tuple:
        raise ValueError("PR history chain collections must be canonical tuples")
    for index, transition in enumerate(chain.transitions):
        _validate_canonical_transition_container(transition, f"transition {index}")
    for index, timing in enumerate(chain.timings):
        if type(timing) is not TransitionTiming or type(timing.coordinate) is not AtomicCoordinate:
            raise ValueError(f"transition timing {index} has a noncanonical type")
        if type(timing.accepted) is not bool:
            raise ValueError(f"transition timing {index} has a noncanonical acceptance flag")
        _validate_canonical_frozen_json(timing.values, f"transition timing {index} values")
        _sha256(timing.timing_sha256, f"transition timing {index} timing_sha256")
    if chain.failed_reference is not None:
        if type(chain.failed_reference) is not FailedReference:
            raise ValueError("PR history chain has a noncanonical failed-reference type")
        _validate_canonical_frozen_json(chain.failed_reference.applied_record, "failed applied record")
        _validate_canonical_frozen_json(chain.failed_reference.reference_record, "failed reference record")
    if type(chain.termination) is not str:
        raise ValueError("PR history chain has a noncanonical termination")
    if chain.prior_chain_sha256 is not None:
        _sha256(chain.prior_chain_sha256, "PR history prior_chain_sha256")
    _sha256(chain.chain_sha256, "PR history chain_sha256")


def _snapshot_coordinate(coordinate: AtomicCoordinate, label: str) -> AtomicCoordinate:
    if (
        type(coordinate) is not AtomicCoordinate
        or type(coordinate.frame) is not int
        or type(coordinate.substep) is not int
    ):
        raise ValueError(f"{label} has a noncanonical coordinate")
    return AtomicCoordinate(coordinate.frame, coordinate.substep)


def _snapshot_committed_state(state: CommittedState, label: str) -> CommittedState:
    if type(state) is not CommittedState:
        raise ValueError(f"{label} has a noncanonical state type")
    snapshot = CommittedState(
        manifest_sha256=_sha256(state.manifest_sha256, f"{label} manifest_sha256"),
        coordinate=_snapshot_coordinate(state.coordinate, f"{label} coordinate"),
        q=state.q,
        qd=state.qd,
        particle_flags=state.particle_flags,
    )
    if snapshot.state_sha256 != _sha256(state.state_sha256, f"{label} state_sha256"):
        raise ValueError(f"{label} raw content changed after authentication")
    return snapshot


def _snapshot_applied_state(state: AppliedAtomicState, label: str) -> AppliedAtomicState:
    if type(state) is not AppliedAtomicState:
        raise ValueError(f"{label} has a noncanonical applied-state type")
    if type(state.callback_applied) is not bool or type(state.action) is not str:
        raise ValueError(f"{label} has noncanonical applied-state scalars")
    if state.schedule_value_name is not None and type(state.schedule_value_name) is not str:
        raise ValueError(f"{label} has a noncanonical schedule-value name")
    if state.schedule_value is not None and type(state.schedule_value) is not float:
        raise ValueError(f"{label} has a noncanonical schedule value")
    snapshot = AppliedAtomicState(
        manifest_sha256=_sha256(state.manifest_sha256, f"{label} manifest_sha256"),
        coordinate=_snapshot_coordinate(state.coordinate, f"{label} coordinate"),
        input_state_sha256=_sha256(state.input_state_sha256, f"{label} input_state_sha256"),
        callback_applied=state.callback_applied,
        action=state.action,
        schedule_value_name=state.schedule_value_name,
        schedule_value=state.schedule_value,
        q=state.q,
        particle_flags=state.particle_flags,
        pinned_indices=state.pinned_indices,
        pin_targets=state.pin_targets,
    )
    if snapshot.applied_sha256 != _sha256(state.applied_sha256, f"{label} applied_sha256"):
        raise ValueError(f"{label} raw content changed after authentication")
    return snapshot


def _snapshot_history_manifest(manifest: PRHistoryManifest) -> PRHistoryManifest:
    if type(manifest) is not PRHistoryManifest:
        raise ValueError("PR history chain has a noncanonical manifest type")
    if type(manifest.kind) is not str or manifest.kind not in _HISTORY_KINDS:
        raise ValueError("PR history manifest has a noncanonical kind")
    if type(manifest.source_path) is not str or type(manifest.source_revision) is not str:
        raise ValueError("PR history manifest has noncanonical source identity")
    if manifest.compression_ratio is not None and type(manifest.compression_ratio) is not float:
        raise ValueError("PR history manifest has a noncanonical compression ratio")
    for name in ("total_frames", "substeps_per_frame", "schema_version"):
        if type(getattr(manifest, name)) is not int:
            raise ValueError(f"PR history manifest has a noncanonical {name}")
    if type(manifest.dt_seconds) is not float:
        raise ValueError("PR history manifest has a noncanonical time step")
    schedule = _thaw_json(_freeze_json(manifest.schedule))
    snapshot = PRHistoryManifest(
        kind=manifest.kind,
        source_path=manifest.source_path,
        base_physical_sha256=_sha256(manifest.base_physical_sha256, "base_physical_sha256"),
        topology_sha256=_sha256(manifest.topology_sha256, "topology_sha256"),
        material_sha256=_sha256(manifest.material_sha256, "material_sha256"),
        compression_ratio=manifest.compression_ratio,
        schedule=schedule,
        total_frames=manifest.total_frames,
        substeps_per_frame=manifest.substeps_per_frame,
        dt_seconds=manifest.dt_seconds,
        schema_version=manifest.schema_version,
        source_revision=manifest.source_revision,
    )
    if snapshot.manifest_sha256 != _sha256(manifest.manifest_sha256, "manifest_sha256"):
        raise ValueError("PR history manifest raw content changed after authentication")
    return snapshot


def _snapshot_checkpoint(checkpoint: HistoryCheckpoint, label: str) -> HistoryCheckpoint:
    if type(checkpoint) is not HistoryCheckpoint:
        raise ValueError(f"{label} has a noncanonical checkpoint type")
    prior = checkpoint.prior_transition_sha256
    if prior is not None:
        prior = _sha256(prior, f"{label} prior_transition_sha256")
    snapshot = HistoryCheckpoint(
        manifest_sha256=_sha256(checkpoint.manifest_sha256, f"{label} manifest_sha256"),
        state=_snapshot_committed_state(checkpoint.state, f"{label} state"),
        prior_transition_sha256=prior,
        prefix_sha256=_sha256(checkpoint.prefix_sha256, f"{label} prefix_sha256"),
    )
    if snapshot.checkpoint_sha256 != _sha256(checkpoint.checkpoint_sha256, f"{label} checkpoint_sha256"):
        raise ValueError(f"{label} raw content changed after authentication")
    return snapshot


def _snapshot_transition(transition: HistoryTransition, label: str) -> HistoryTransition:
    if type(transition) is not HistoryTransition or "as_dict" in vars(transition):
        raise ValueError(f"{label} has a noncanonical transition type")
    applied_record = _thaw_json(_freeze_json(transition.applied_record))
    reference_record = _thaw_json(_freeze_json(transition.reference_record))
    snapshot = HistoryTransition(
        manifest_sha256=_sha256(transition.manifest_sha256, f"{label} manifest_sha256"),
        coordinate=_snapshot_coordinate(transition.coordinate, f"{label} coordinate"),
        next_coordinate=_snapshot_coordinate(transition.next_coordinate, f"{label} next_coordinate"),
        input_state_sha256=_sha256(transition.input_state_sha256, f"{label} input_state_sha256"),
        input_prefix_sha256=_sha256(transition.input_prefix_sha256, f"{label} input_prefix_sha256"),
        input_state=_snapshot_committed_state(transition.input_state, f"{label} input"),
        applied_state=_snapshot_applied_state(transition.applied_state, f"{label} applied"),
        applied_record=applied_record,
        scene_sha256=_sha256(transition.scene_sha256, f"{label} scene_sha256"),
        objective_instance_sha256=_sha256(
            transition.objective_instance_sha256,
            f"{label} objective_instance_sha256",
        ),
        dt_seconds=transition.dt_seconds,
        topology_sha256=_sha256(transition.topology_sha256, f"{label} topology_sha256"),
        material_sha256=_sha256(transition.material_sha256, f"{label} material_sha256"),
        inertial_target=transition.inertial_target,
        reference_record=reference_record,
        reference_positions=transition.reference_positions,
        output_state=_snapshot_committed_state(transition.output_state, f"{label} output"),
    )
    if snapshot.transition_sha256 != _sha256(transition.transition_sha256, f"{label} transition_sha256"):
        raise ValueError(f"{label} raw content changed after authentication")
    return snapshot


def _snapshot_chain(chain: PRHistoryChain) -> PRHistoryChain:
    """Traverse caller-owned history content once into canonical dataclasses."""
    if type(chain) is not PRHistoryChain or "verify" in vars(chain) or "checkpoint_at" in vars(chain):
        raise ValueError("correction evidence requires the canonical PRHistoryChain implementation")
    if type(chain.transitions) is not tuple or type(chain.timings) is not tuple:
        raise ValueError("PR history chain collections must be canonical tuples")
    manifest = _snapshot_history_manifest(chain.manifest)
    transitions = tuple(
        _snapshot_transition(item, f"transition {index}") for index, item in enumerate(chain.transitions)
    )
    timings = []
    for index, timing in enumerate(chain.timings):
        if type(timing) is not TransitionTiming or type(timing.accepted) is not bool:
            raise ValueError(f"transition timing {index} has a noncanonical type")
        timing_snapshot = TransitionTiming(
            coordinate=_snapshot_coordinate(timing.coordinate, f"transition timing {index} coordinate"),
            accepted=timing.accepted,
            values=_thaw_json(_freeze_json(timing.values)),
        )
        if timing_snapshot.timing_sha256 != _sha256(timing.timing_sha256, f"transition timing {index} SHA-256"):
            raise ValueError(f"transition timing {index} raw content changed after authentication")
        timings.append(timing_snapshot)
    failure = chain.failed_reference
    failure_snapshot = None
    if failure is not None:
        if type(failure) is not FailedReference:
            raise ValueError("PR history chain has a noncanonical failed-reference type")
        if type(failure.failures) is not tuple or any(type(item) is not str for item in failure.failures):
            raise ValueError("failed reference has noncanonical failures")
        failure_snapshot = FailedReference(
            manifest_sha256=_sha256(failure.manifest_sha256, "failed manifest_sha256"),
            coordinate=_snapshot_coordinate(failure.coordinate, "failed coordinate"),
            input_state_sha256=_sha256(failure.input_state_sha256, "failed input_state_sha256"),
            input_prefix_sha256=_sha256(failure.input_prefix_sha256, "failed input_prefix_sha256"),
            applied_record=_thaw_json(_freeze_json(failure.applied_record)),
            scene_sha256=_sha256(failure.scene_sha256, "failed scene_sha256"),
            objective_instance_sha256=_sha256(
                failure.objective_instance_sha256,
                "failed objective_instance_sha256",
            ),
            reference_record=_thaw_json(_freeze_json(failure.reference_record)),
            failures=failure.failures,
        )
        if failure_snapshot.failure_sha256 != _sha256(failure.failure_sha256, "failed failure_sha256"):
            raise ValueError("failed reference raw content changed after authentication")
    if type(chain.termination) is not str:
        raise ValueError("PR history chain has a noncanonical termination")
    prior = chain.prior_chain_sha256
    if prior is not None:
        prior = _sha256(prior, "prior_chain_sha256")
    snapshot = PRHistoryChain(
        manifest=manifest,
        initial_checkpoint=_snapshot_checkpoint(chain.initial_checkpoint, "chain initial checkpoint"),
        transitions=transitions,
        timings=tuple(timings),
        final_checkpoint=_snapshot_checkpoint(chain.final_checkpoint, "chain final checkpoint"),
        termination=chain.termination,
        failed_reference=failure_snapshot,
        prior_chain_sha256=prior,
    )
    if snapshot.chain_sha256 != _sha256(chain.chain_sha256, "PR history chain_sha256"):
        raise ValueError("PR history chain SHA-256 changed after authentication")
    return snapshot


def _chain_member_ordinal(chain: PRHistoryChain, transition: HistoryTransition) -> int:
    if type(chain) is not PRHistoryChain or type(chain.transitions) is not tuple:
        raise ValueError("correction evidence requires a canonical tuple-backed PR history chain")
    if type(transition) is not HistoryTransition or type(transition.coordinate) is not AtomicCoordinate:
        raise ValueError("correction evidence requires a canonical HistoryTransition")
    ordinal = transition.coordinate.ordinal
    if ordinal >= len(chain.transitions) or chain.transitions[ordinal] is not transition:
        raise ValueError("transition must be the exact object stored at its chain ordinal")
    return ordinal


def _verify_history_chain_raw_content(history: PRSceneHistory, chain: PRHistoryChain) -> None:
    """Defend against reversible NumPy write flags in history containers."""
    _validate_canonical_chain_container(chain)
    _verify_self_hashed_record(history.manifest.as_dict(), "manifest_sha256", "PR history manifest")
    _verify_self_hashed_record(history.static_bundle.as_dict(), "static_sha256", "PR history static bundle")
    _verify_self_hashed_record(
        history.initial_checkpoint.state.as_dict(),
        "state_sha256",
        "canonical PR root state",
    )
    _verify_self_hashed_record(
        history.initial_checkpoint.as_dict(),
        "checkpoint_sha256",
        "canonical PR root checkpoint",
    )
    for label, checkpoint in (
        ("chain initial checkpoint", chain.initial_checkpoint),
        ("chain final checkpoint", chain.final_checkpoint),
    ):
        _verify_self_hashed_record(checkpoint.state.as_dict(), "state_sha256", f"{label} state")
        _verify_self_hashed_record(checkpoint.as_dict(), "checkpoint_sha256", label)
    for transition in chain.transitions:
        ordinal = transition.coordinate.ordinal
        _verify_self_hashed_record(
            transition.input_state.as_dict(),
            "state_sha256",
            f"transition {ordinal} input state",
        )
        _verify_self_hashed_record(
            transition.applied_state.as_dict(),
            "applied_sha256",
            f"transition {ordinal} applied state",
        )
        _verify_self_hashed_record(
            transition.output_state.as_dict(),
            "state_sha256",
            f"transition {ordinal} output state",
        )
        _verify_self_hashed_record(
            transition.as_dict(),
            "transition_sha256",
            f"transition {ordinal}",
        )
    if chain.failed_reference is not None:
        _verify_self_hashed_record(
            chain.failed_reference.as_dict(),
            "failure_sha256",
            "failed reference",
        )
    chain_payload = {
        "manifest_sha256": chain.manifest.manifest_sha256,
        "initial_checkpoint_sha256": chain.initial_checkpoint.checkpoint_sha256,
        "transition_sha256": [item.transition_sha256 for item in chain.transitions],
        "final_checkpoint_sha256": chain.final_checkpoint.checkpoint_sha256,
        "termination": chain.termination,
        "failed_reference_sha256": (None if chain.failed_reference is None else chain.failed_reference.failure_sha256),
        "prior_chain_sha256": chain.prior_chain_sha256,
    }
    _sha256(chain.chain_sha256, "PR history chain_sha256")
    if chain.chain_sha256 != _canonical_digest(chain_payload):
        raise ValueError("PR history chain SHA-256 changed after authentication")


def _reconstruct_canonical_history(history: PRSceneHistory) -> PRSceneHistory:
    """Replace caller behavior and mutable private state with a fresh history."""
    if type(history) is not PRSceneHistory or any(callable(value) for value in vars(history).values()):
        raise ValueError("correction evidence requires an unmodified canonical PRSceneHistory instance")
    if type(history.kind) is not str or history.kind not in _HISTORY_KINDS:
        raise ValueError("correction evidence requires a canonical PR history kind")
    _validate_canonical_history_manifest(history.manifest)
    if history.kind != history.manifest.kind:
        raise ValueError("PR history kind disagrees with its manifest")
    supplied_root = _snapshot_chain(
        PRHistoryChain(
            manifest=history.manifest,
            initial_checkpoint=history.initial_checkpoint,
            transitions=(),
            timings=(),
            final_checkpoint=history.initial_checkpoint,
            termination="range_complete",
        )
    )
    _verify_history_chain_raw_content(history, supplied_root)
    canonical = PRSceneHistory(history.manifest.kind)
    _verify_history_chain_raw_content(
        canonical,
        PRHistoryChain(
            manifest=canonical.manifest,
            initial_checkpoint=canonical.initial_checkpoint,
            transitions=(),
            timings=(),
            final_checkpoint=canonical.initial_checkpoint,
            termination="range_complete",
        ),
    )
    for label, supplied, expected in (
        ("manifest", history.manifest.as_dict(), canonical.manifest.as_dict()),
        ("static bundle", history.static_bundle.as_dict(), canonical.static_bundle.as_dict()),
        ("root checkpoint", history.initial_checkpoint.as_dict(), canonical.initial_checkpoint.as_dict()),
        ("base scene", history._base_scene.manifest(), canonical._base_scene.manifest()),
    ):
        if supplied != expected:
            raise ValueError(f"PR history {label} differs from fresh canonical reconstruction")
    return canonical


def _verify_transition_by_canonical_replay(
    history: PRSceneHistory,
    chain: PRHistoryChain,
    transition: HistoryTransition,
) -> None:
    """Regenerate the complete canonical prefix through one transition."""
    ordinal = transition.coordinate.ordinal
    if ordinal >= len(chain.transitions) or chain.transitions[ordinal].as_dict() != transition.as_dict():
        raise ValueError("canonical root chain does not contain the transition at its ordinal")
    replay = PRSceneHistory.generate(
        history,
        stop=transition.next_coordinate,
        max_transitions=ordinal + 1,
    )
    expected_termination = (
        "history_complete" if transition.next_coordinate == history.manifest.end_coordinate else "range_complete"
    )
    expected_prefix = tuple(item.as_dict() for item in chain.transitions[: ordinal + 1])
    actual_prefix = tuple(item.as_dict() for item in replay.transitions)
    if (
        replay.termination != expected_termination
        or len(replay.transitions) != ordinal + 1
        or actual_prefix != expected_prefix
        or replay.final_checkpoint.as_dict()
        != PRHistoryChain.checkpoint_at(chain, transition.next_coordinate).as_dict()
    ):
        raise ValueError("transition prefix does not match canonical root dense-reference replay")


def _validated_canonical_problem(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
) -> tuple[NewtonProblem, str, str]:
    if type(scene) is not TetBenchmarkScene or any(callable(value) for value in vars(scene).values()):
        raise ValueError("benchmark validation requires the canonical TetBenchmarkScene implementation")
    if type(problem) is not NewtonProblem:
        raise ValueError("Newton problem validation requires the canonical NewtonProblem type")
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    expected_problem = build_common_problem(scene)
    expected_objective = common_objective_manifest(scene, expected_problem)
    actual_objective = common_objective_manifest(scene, problem)
    objective_instance_sha256 = str(actual_objective["objective_instance_sha256"])
    if objective_instance_sha256 != expected_objective["objective_instance_sha256"]:
        raise ValueError("Newton problem does not match the supplied scene")
    return expected_problem, scene_sha256, objective_instance_sha256


def _validated_problem_identity(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
) -> tuple[str, str]:
    _, scene_sha256, objective_instance_sha256 = _validated_canonical_problem(scene, problem)
    return scene_sha256, objective_instance_sha256


def _metrics_equal(left: CommonStateMetrics, right: CommonStateMetrics) -> bool:
    return _canonical_digest(left.as_dict()) == _canonical_digest(right.as_dict())


def _validate_metrics_against_context(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    positions: np.ndarray | torch.Tensor,
    reference_positions: np.ndarray | torch.Tensor,
    metrics: CommonStateMetrics,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Re-evaluate claimed metrics from the bound raw state and objective."""
    if type(metrics) is not CommonStateMetrics:
        raise ValueError(f"{label} metrics require the canonical CommonStateMetrics type")
    canonical_problem, scene_sha256, objective_instance_sha256 = _validated_canonical_problem(scene, problem)
    candidate = _readonly_positions(positions, f"{label} positions")
    reference = _readonly_positions(reference_positions, "accepted reference positions")
    if candidate.shape != reference.shape:
        raise ValueError(f"{label} positions do not match the accepted reference")
    measured = evaluate_common_state(canonical_problem, candidate, reference_positions=reference)
    if not _metrics_equal(metrics, measured):
        raise ValueError(f"{label} metrics disagree with independent objective evaluation")
    return candidate, reference, scene_sha256, objective_instance_sha256


def _vbd_state_provenance(result: VBDRunResult) -> dict[str, object]:
    provenance = {
        "contract": "pss-solver-vbd-state-evidence-v1",
        "iterations": result.iterations,
        "requested_tile_solve": result.requested_tile_solve,
        "effective_tile_solve": result.effective_tile_solve,
        "color_group_count": result.color_group_count,
        "device": result.device,
        "physical_state_sha256": result.physical_state_sha256,
        "iterate_zero_sha256": result.iterate_zero_sha256,
        "result_state_sha256": result.result_state_sha256,
        "velocities_sha256": _array_digest(result.velocities),
    }
    provenance["vbd_execution_sha256"] = _canonical_digest(provenance)
    return provenance


def _vbd_deterministic_execution_record(result: VBDRunResult) -> dict[str, object]:
    return {
        "positions_sha256": _array_digest(result.positions),
        "velocities_sha256": _array_digest(result.velocities),
        "iterations": result.iterations,
        "requested_tile_solve": result.requested_tile_solve,
        "effective_tile_solve": result.effective_tile_solve,
        "color_group_count": result.color_group_count,
        "device": result.device,
        "scene_sha256": result.scene_sha256,
        "objective_instance_sha256": result.objective_instance_sha256,
        "physical_state_sha256": result.physical_state_sha256,
        "iterate_zero_sha256": result.iterate_zero_sha256,
        "result_state_sha256": result.result_state_sha256,
    }


def _validate_vbd_run_by_replay(scene: TetBenchmarkScene, result: VBDRunResult) -> None:
    """Authenticate deterministic CPU VBD content independently of timings."""
    if type(result) is not VBDRunResult:
        raise ValueError("VBD correction evidence requires the canonical VBDRunResult type")
    if type(result.iterations) is not int or result.iterations < 1:
        raise ValueError("VBD correction evidence has an invalid iteration count")
    if type(result.requested_tile_solve) is not bool or type(result.effective_tile_solve) is not bool:
        raise ValueError("VBD correction evidence has invalid tile-solve flags")
    if type(result.color_group_count) is not int or result.color_group_count < 1:
        raise ValueError("VBD correction evidence has an invalid color-group count")
    if type(result.device) is not str or result.device != "cpu":
        raise ValueError("correction-ceiling VBD evidence requires deterministic scalar CPU execution")
    if result.requested_tile_solve or result.effective_tile_solve:
        raise ValueError("correction-ceiling VBD evidence requires deterministic scalar CPU execution")
    for name in ("setup_seconds", "warmup_seconds"):
        value = getattr(result, name)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"VBD correction evidence has invalid {name}")
    if type(result.repeat_seconds) is not tuple or type(result.transfer_seconds) is not tuple:
        raise ValueError("VBD correction evidence timing sequences must be tuples")
    if not result.repeat_seconds or len(result.repeat_seconds) != len(result.transfer_seconds):
        raise ValueError("VBD correction evidence timing sequences are incomplete")
    if any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in (*result.repeat_seconds, *result.transfer_seconds)
    ):
        raise ValueError("VBD correction evidence timings must be finite non-negative floats")
    for name in (
        "scene_sha256",
        "objective_instance_sha256",
        "physical_state_sha256",
        "iterate_zero_sha256",
        "result_state_sha256",
        "run_sha256",
    ):
        _sha256(getattr(result, name), f"VBD correction evidence {name}")
    if result.run_sha256 != _vbd_run_digest(result):
        raise ValueError("VBD correction evidence run SHA-256 verification failed")
    replay = run_vbd(
        scene,
        result.iterations,
        device="cpu",
        tile_solve=False,
        warmup=False,
        repeats=1,
    )
    if _canonical_digest(_vbd_deterministic_execution_record(result)) != _canonical_digest(
        _vbd_deterministic_execution_record(replay)
    ):
        raise ValueError("VBD correction evidence does not match deterministic CPU replay")


@dataclasses.dataclass(frozen=True, eq=False)
class CorrectionStart:
    """Immutable, objective-bound correction initializer or comparator."""

    role: str
    positions: np.ndarray
    metrics: CommonStateMetrics
    scene_sha256: str
    objective_instance_sha256: str
    reference_position_sha256: str
    provenance: Mapping[str, object]
    _validation_scene: TetBenchmarkScene = dataclasses.field(repr=False, compare=False)
    _validation_problem: NewtonProblem = dataclasses.field(repr=False, compare=False)
    _validation_reference_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    _validation_vbd_result: VBDRunResult | None = dataclasses.field(default=None, repr=False, compare=False)
    position_sha256: str = dataclasses.field(init=False)
    evidence_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.role) is not str or not self.role:
            raise ValueError("correction start role must be nonempty")
        for name, value in (
            ("scene_sha256", self.scene_sha256),
            ("objective_instance_sha256", self.objective_instance_sha256),
            ("reference_position_sha256", self.reference_position_sha256),
        ):
            _sha256(value, name)
        positions, reference, actual_scene_sha256, actual_objective_sha256 = _validate_metrics_against_context(
            self._validation_scene,
            self._validation_problem,
            self.positions,
            self._validation_reference_positions,
            self.metrics,
            label="correction start",
        )
        if self.scene_sha256 != actual_scene_sha256:
            raise ValueError("correction start scene identity disagrees with its validation context")
        if self.objective_instance_sha256 != actual_objective_sha256:
            raise ValueError("correction start objective identity disagrees with its validation context")
        if self.reference_position_sha256 != _array_digest(reference):
            raise ValueError("correction start reference identity disagrees with its validation context")
        position_sha256 = _array_digest(positions)
        if not _finite_metrics(self.metrics):
            raise ValueError("correction start metrics must be finite")
        if self.metrics.position_sha256 != position_sha256:
            raise ValueError("correction start metrics do not describe its positions")
        if self.metrics.free_rms_error_m is None or self.metrics.mass_weighted_rms_error_m is None:
            raise ValueError("correction start metrics must be evaluated against the accepted reference")
        provenance = _thaw_json(_freeze_json(self.provenance))
        if not isinstance(provenance, Mapping):
            raise ValueError("correction start provenance must be a mapping")
        is_vbd_role = self.role.startswith("vbd-k")
        is_vbd_provenance = provenance.get("contract") == "pss-solver-vbd-state-evidence-v1"
        if is_vbd_role != is_vbd_provenance:
            raise ValueError("reserved vbd-k roles require the VBD evidence contract and vice versa")
        if is_vbd_provenance:
            result = self._validation_vbd_result
            if result is None:
                raise ValueError("VBD correction evidence requires its original run result")
            _validate_vbd_run_by_replay(self._validation_scene, result)
            if self.role != f"vbd-k{result.iterations}":
                raise ValueError("VBD correction evidence role disagrees with its replayed iteration count")
            if (
                result.scene_sha256 != self.scene_sha256
                or result.objective_instance_sha256 != self.objective_instance_sha256
            ):
                raise ValueError("VBD correction evidence run belongs to another objective")
            if _array_digest(result.positions) != position_sha256:
                raise ValueError("VBD correction evidence positions changed")
            if _canonical_digest(provenance) != _canonical_digest(_vbd_state_provenance(result)):
                raise ValueError("VBD correction evidence provenance changed from its original run")
        elif self._validation_vbd_result is not None:
            raise ValueError("non-VBD correction evidence cannot carry a VBD run result")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "_validation_reference_positions", reference)
        object.__setattr__(self, "provenance", _freeze_json(provenance))
        object.__setattr__(self, "position_sha256", position_sha256)
        object.__setattr__(self, "evidence_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _STATE_CONTRACT,
            "role": self.role,
            "position_sha256": self.position_sha256,
            "metrics": self.metrics.as_dict(),
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "reference_position_sha256": self.reference_position_sha256,
            "provenance": _thaw_json(self.provenance),
        }

    def as_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["evidence_sha256"] = self.evidence_sha256
        return payload


def build_correction_start(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    positions: np.ndarray | torch.Tensor,
    reference_positions: np.ndarray | torch.Tensor,
    *,
    role: str,
    provenance: Mapping[str, object],
    _validation_vbd_result: VBDRunResult | None = None,
) -> CorrectionStart:
    """Build generic state evidence on one verified common objective."""
    canonical_problem, scene_sha256, objective_instance_sha256 = _validated_canonical_problem(scene, problem)
    candidate = _readonly_positions(positions, "correction start positions")
    reference = _readonly_positions(reference_positions, "reference positions")
    if candidate.shape != reference.shape:
        raise ValueError("reference positions do not match the correction start")
    metrics = evaluate_common_state(canonical_problem, candidate, reference_positions=reference)
    return CorrectionStart(
        role=role,
        positions=candidate,
        metrics=metrics,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        reference_position_sha256=_array_digest(reference),
        provenance=provenance,
        _validation_scene=scene,
        _validation_problem=problem,
        _validation_reference_positions=reference,
        _validation_vbd_result=_validation_vbd_result,
    )


def build_vbd_correction_start(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    result: VBDRunResult,
    reference_positions: np.ndarray | torch.Tensor,
) -> CorrectionStart:
    """Bind one fresh VBD run as correction-state evidence."""
    canonical_problem, scene_sha256, objective_instance_sha256 = _validated_canonical_problem(scene, problem)
    if result.scene_sha256 != scene_sha256:
        raise ValueError("VBD result belongs to a different scene")
    if result.objective_instance_sha256 != objective_instance_sha256:
        raise ValueError("VBD result belongs to a different objective")
    if result.result_state_sha256 != _array_digest(result.positions):
        raise ValueError("VBD result-state hash does not match its positions")
    if result.physical_state_sha256 != _array_digest(scene.x_current):
        raise ValueError("VBD physical-state hash does not match the scene")
    iterate_zero = canonical_problem.inertial_target.index_copy(
        0,
        canonical_problem.pinned,
        canonical_problem.pin_targets,
    ).numpy()
    if result.iterate_zero_sha256 != _array_digest(iterate_zero):
        raise ValueError("VBD iterate-zero hash does not match the common objective")
    if result.run_sha256 != _vbd_run_digest(result):
        raise ValueError("VBD run SHA-256 verification failed")
    provenance = _vbd_state_provenance(result)
    return build_correction_start(
        scene,
        canonical_problem,
        result.positions,
        reference_positions,
        role=f"vbd-k{result.iterations}",
        provenance=provenance,
        _validation_vbd_result=result,
    )


def _vbd_timing_record(result: VBDRunResult, evidence: CorrectionStart) -> dict[str, object]:
    if result.run_sha256 != _vbd_run_digest(result):
        raise ValueError("VBD run SHA-256 verification failed")
    payload = {
        "contract": "pss-solver-vbd-timing-v1",
        "performance_evidence": False,
        "measurement_provenance": _DIAGNOSTIC_TIMING_PROVENANCE,
        "state_evidence_sha256": evidence.evidence_sha256,
        "run_sha256": result.run_sha256,
        "setup_seconds": result.setup_seconds,
        "warmup_seconds": result.warmup_seconds,
        "repeat_seconds": list(result.repeat_seconds),
        "transfer_seconds": list(result.transfer_seconds),
    }
    payload["timing_sha256"] = _canonical_digest(payload)
    _validate_vbd_timing_record(payload, evidence)
    return payload


def _validate_vbd_state_evidence(
    evidence: CorrectionStart,
    *,
    role: str,
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
) -> Mapping[str, object]:
    provenance = _thaw_json(evidence.provenance)
    if not isinstance(provenance, Mapping) or provenance.get("contract") != "pss-solver-vbd-state-evidence-v1":
        raise ValueError(f"{role} evidence has the wrong VBD provenance contract")
    expected_iterations = int(role.removeprefix("vbd-k"))
    if provenance.get("iterations") != expected_iterations:
        raise ValueError(f"{role} evidence has the wrong iteration count")
    for name in ("requested_tile_solve", "effective_tile_solve"):
        if not isinstance(provenance.get(name), bool):
            raise ValueError(f"{role} evidence has an invalid {name}")
    color_count = provenance.get("color_group_count")
    if isinstance(color_count, bool) or not isinstance(color_count, int) or color_count < 1:
        raise ValueError(f"{role} evidence has an invalid color-group count")
    if not isinstance(provenance.get("device"), str) or not provenance["device"]:
        raise ValueError(f"{role} evidence has an invalid device")
    for name in (
        "physical_state_sha256",
        "iterate_zero_sha256",
        "result_state_sha256",
        "velocities_sha256",
        "vbd_execution_sha256",
    ):
        _sha256(provenance.get(name), f"{role} {name}")
    if provenance["physical_state_sha256"] != _array_digest(scene.x_current):
        raise ValueError(f"{role} physical-state hash changed")
    iterate_zero = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets).numpy()
    if provenance["iterate_zero_sha256"] != _array_digest(iterate_zero):
        raise ValueError(f"{role} iterate-zero hash changed")
    if provenance["result_state_sha256"] != evidence.position_sha256:
        raise ValueError(f"{role} result-state hash changed")
    execution = dict(provenance)
    supplied_execution_sha256 = execution.pop("vbd_execution_sha256")
    if supplied_execution_sha256 != _canonical_digest(execution):
        raise ValueError(f"{role} execution SHA-256 verification failed")
    return provenance


def _validate_vbd_timing_record(timing: Mapping[str, object], evidence: CorrectionStart) -> None:
    if timing.get("contract") != "pss-solver-vbd-timing-v1":
        raise ValueError("VBD timing evidence has the wrong contract")
    if (
        timing.get("performance_evidence") is not False
        or timing.get("measurement_provenance") != _DIAGNOSTIC_TIMING_PROVENANCE
    ):
        raise ValueError("VBD timing must remain self-reported diagnostic data")
    if timing.get("state_evidence_sha256") != evidence.evidence_sha256:
        raise ValueError("VBD timing evidence uses the wrong state evidence")
    for name in ("setup_seconds", "warmup_seconds"):
        value = timing.get(name)
        if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"VBD timing {name} must be a finite non-negative float")
    repeats = timing.get("repeat_seconds")
    transfers = timing.get("transfer_seconds")
    if (
        not isinstance(repeats, (list, tuple))
        or not isinstance(transfers, (list, tuple))
        or not repeats
        or len(repeats) != len(transfers)
    ):
        raise ValueError("VBD timing repeats are incomplete")
    if any(not isinstance(value, float) or not math.isfinite(value) or value < 0.0 for value in (*repeats, *transfers)):
        raise ValueError("VBD repeat timings must be finite non-negative floats")
    provenance = _thaw_json(evidence.provenance)
    run_payload = {
        "positions_sha256": evidence.position_sha256,
        "velocities_sha256": provenance.get("velocities_sha256"),
        "iterations": provenance.get("iterations"),
        "requested_tile_solve": provenance.get("requested_tile_solve"),
        "effective_tile_solve": provenance.get("effective_tile_solve"),
        "color_group_count": provenance.get("color_group_count"),
        "device": provenance.get("device"),
        "setup_seconds": timing["setup_seconds"],
        "warmup_seconds": timing["warmup_seconds"],
        "repeat_seconds": list(repeats),
        "transfer_seconds": list(transfers),
        "scene_sha256": evidence.scene_sha256,
        "objective_instance_sha256": evidence.objective_instance_sha256,
        "physical_state_sha256": provenance.get("physical_state_sha256"),
        "iterate_zero_sha256": provenance.get("iterate_zero_sha256"),
        "result_state_sha256": provenance.get("result_state_sha256"),
    }
    run_sha256 = timing.get("run_sha256")
    _sha256(run_sha256, "VBD run_sha256")
    if run_sha256 != _canonical_digest(run_payload):
        raise ValueError("VBD timing run SHA-256 is inconsistent with its execution evidence")
    timing_payload = dict(_thaw_json(timing))
    timing_sha256 = timing_payload.pop("timing_sha256", None)
    if timing_sha256 != _canonical_digest(timing_payload):
        raise ValueError("VBD timing SHA-256 verification failed")
    result = evidence._validation_vbd_result
    if result is None or result.run_sha256 != _vbd_run_digest(result):
        raise ValueError("VBD timing evidence lost its original run result")
    expected = {
        "contract": "pss-solver-vbd-timing-v1",
        "performance_evidence": False,
        "measurement_provenance": _DIAGNOSTIC_TIMING_PROVENANCE,
        "state_evidence_sha256": evidence.evidence_sha256,
        "run_sha256": result.run_sha256,
        "setup_seconds": result.setup_seconds,
        "warmup_seconds": result.warmup_seconds,
        "repeat_seconds": list(result.repeat_seconds),
        "transfer_seconds": list(result.transfer_seconds),
    }
    expected["timing_sha256"] = _canonical_digest(expected)
    if _thaw_json(timing) != expected:
        raise ValueError("VBD timing evidence changed from its original run result")


def _validate_solver_record(
    record: Mapping[str, object],
    config: NewtonResidualPolishConfig,
    correction_budget: int,
) -> None:
    trace = record.get("trace")
    if not isinstance(trace, tuple) or not trace:
        raise ValueError("residual-Newton record must contain a nonempty trace")
    accepted = record.get("accepted_iterations")
    if isinstance(accepted, bool) or not isinstance(accepted, int) or not 0 <= accepted <= correction_budget:
        raise ValueError("residual-Newton applied correction count is invalid")
    if len(trace) != accepted + 1:
        raise ValueError("residual-Newton trace length disagrees with applied corrections")
    if any(
        not isinstance(item, Mapping) or type(item.get("iteration")) is not int or item["iteration"] != index
        for index, item in enumerate(trace)
    ):
        raise ValueError("residual-Newton trace indices are invalid")

    residual_scale = record.get("residual_scale")
    gradient_limit = record.get("gradient_limit")
    if not isinstance(residual_scale, float) or not math.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("residual-Newton residual scale is invalid")
    if type(gradient_limit) is not float or gradient_limit != 0.0:
        raise ValueError("zero-tolerance correction prefixes require a zero gradient limit")

    accepted_trials = 0
    eps = np.finfo(np.float64).eps
    for index, item in enumerate(trace):
        if not isinstance(item, Mapping):
            raise ValueError("residual-Newton trace entries must be mappings")
        gradient = item.get("gradient_norm")
        relative = item.get("relative_residual")
        merit = item.get("residual_merit")
        objective = item.get("objective")
        scalars = (gradient, relative, merit, objective)
        if any(not isinstance(value, float) or not math.isfinite(value) for value in scalars):
            raise ValueError("residual-Newton trace scalars must be finite floats")
        roundoff = _OBJECTIVE_ROUNDOFF_FACTOR * eps
        expected_relative = gradient / residual_scale
        expected_merit = 0.5 * gradient * gradient
        if abs(relative - expected_relative) > roundoff * max(1.0, abs(relative), abs(expected_relative)):
            raise ValueError("residual-Newton trace relative residual is inconsistent")
        if abs(merit - expected_merit) > roundoff * max(1.0, abs(merit), abs(expected_merit)):
            raise ValueError("residual-Newton trace merit is inconsistent")

        step_size = item.get("accepted_step_size")
        step_norm = item.get("accepted_step_norm")
        if not isinstance(step_size, float) or not isinstance(step_norm, float):
            raise ValueError("residual-Newton accepted-step fields must be floats")
        if index == len(trace) - 1:
            if step_size != 0.0 or step_norm != 0.0:
                raise ValueError("residual-Newton final trace item must not have an outgoing step")
            continue
        if not math.isfinite(step_size) or not 0.0 < step_size <= 1.0:
            raise ValueError("residual-Newton accepted step size is invalid")
        if not math.isfinite(step_norm) or step_norm <= 0.0:
            raise ValueError("residual-Newton accepted step norm is invalid")
        exponent = None
        configured_step = 1.0
        for trial in range(config.max_line_search_steps):
            if step_size == configured_step:
                exponent = trial
                break
            configured_step *= config.backtrack
        if exponent is None:
            raise ValueError("residual-Newton accepted step is outside the configured backtrack sequence")
        accepted_trials += exponent + 1
        directional = item.get("merit_directional_derivative")
        minimum_eigenvalue = item.get("hessian_minimum_eigenvalue")
        maximum_eigenvalue = item.get("hessian_maximum_eigenvalue")
        if not isinstance(directional, float) or not math.isfinite(directional) or directional >= 0.0:
            raise ValueError("residual-Newton accepted direction must be finite and descending")
        if (
            not isinstance(minimum_eigenvalue, float)
            or not isinstance(maximum_eigenvalue, float)
            or not math.isfinite(minimum_eigenvalue)
            or not math.isfinite(maximum_eigenvalue)
            or minimum_eigenvalue <= 0.0
            or maximum_eigenvalue < minimum_eigenvalue
        ):
            raise ValueError("residual-Newton accepted Hessian must be finite SPD")
        after = trace[index + 1]
        after_gradient = after.get("gradient_norm")
        after_merit = after.get("residual_merit")
        after_objective = after.get("objective")
        if not isinstance(after_gradient, float) or after_gradient >= gradient:
            raise ValueError("residual-Newton accepted correction must strictly lower the residual")
        armijo_rhs = merit + config.armijo * step_size * directional
        if after_merit > armijo_rhs:
            raise ValueError("residual-Newton accepted correction violates Armijo")
        objective_guard = roundoff * max(1.0, abs(objective), abs(after_objective))
        if after_objective > objective + objective_guard:
            raise ValueError("residual-Newton accepted correction increases the common objective")

    reason = record.get("reason")
    converged = record.get("converged")
    if reason not in _RESIDUAL_POLISH_REASONS:
        raise ValueError("residual-Newton termination reason is invalid")
    if not isinstance(converged, bool):
        raise ValueError("residual-Newton convergence flag must be boolean")
    if converged != (reason == "gradient"):
        raise ValueError("residual-Newton convergence and reason are inconsistent")
    final = trace[-1]
    for name, trace_name in (
        ("final_objective", "objective"),
        ("final_gradient_norm", "gradient_norm"),
        ("final_relative_residual", "relative_residual"),
    ):
        if type(record.get(name)) is not float or record[name] != final.get(trace_name):
            raise ValueError(f"residual-Newton {name} summary is inconsistent")
    if reason == "gradient" and final["gradient_norm"] > gradient_limit:
        raise ValueError("residual-Newton gradient termination exceeds its configured limit")
    if reason == "max_iterations" and accepted != correction_budget:
        raise ValueError("max-iteration result did not apply the exact correction budget")
    work = record.get("work")
    if not isinstance(work, Mapping):
        raise ValueError("residual-Newton work record is missing")
    names = (
        "objective_evaluations",
        "gradient_evaluations",
        "hessian_evaluations",
        "eigenvalue_evaluations",
        "factorization_attempts",
        "line_search_trials",
    )
    if any(isinstance(work.get(name), bool) or not isinstance(work.get(name), int) or work[name] < 0 for name in names):
        raise ValueError("residual-Newton work counters must be non-negative integers")
    if work["objective_evaluations"] != work["gradient_evaluations"]:
        raise ValueError("residual-Newton objective/gradient work is inconsistent")
    if work["objective_evaluations"] != len(trace) + work["line_search_trials"]:
        raise ValueError("residual-Newton evaluation work is inconsistent")
    if not (accepted <= work["hessian_evaluations"] <= accepted + 1):
        raise ValueError("residual-Newton Hessian work is inconsistent")
    if not (work["eigenvalue_evaluations"] <= work["hessian_evaluations"] <= work["eigenvalue_evaluations"] + 1):
        raise ValueError("residual-Newton eigenvalue work is inconsistent")
    if not (accepted <= work["factorization_attempts"] <= work["eigenvalue_evaluations"]):
        raise ValueError("residual-Newton factorization work is inconsistent")
    if work["line_search_trials"] < accepted:
        raise ValueError("residual-Newton line-search work is inconsistent")
    if reason in ("max_iterations", "gradient"):
        if not (
            work["hessian_evaluations"] == work["eigenvalue_evaluations"] == work["factorization_attempts"] == accepted
        ):
            raise ValueError("successful residual-Newton derivative work is inconsistent")
        if work["line_search_trials"] != accepted_trials:
            raise ValueError("successful residual-Newton line-search work is inconsistent")


def _validate_solver_timing(timing: Mapping[str, object], trace_length: int) -> None:
    names = (
        "total_seconds",
        "objective_gradient_seconds",
        "hessian_seconds",
        "linear_solve_seconds",
        "line_search_seconds",
    )
    if any(
        not isinstance(timing.get(name), float) or not math.isfinite(timing[name]) or timing[name] < 0.0
        for name in names
    ):
        raise ValueError("residual-Newton timing values must be finite and non-negative")
    elapsed = timing.get("trace_elapsed_seconds")
    if not isinstance(elapsed, tuple) or len(elapsed) != trace_length:
        raise ValueError("residual-Newton trace timing length is invalid")
    if any(not isinstance(value, float) or not math.isfinite(value) or value < 0.0 for value in elapsed):
        raise ValueError("residual-Newton trace timings must be finite and non-negative")
    if any(after < before for before, after in itertools.pairwise(elapsed)):
        raise ValueError("residual-Newton trace timings must be monotone")


def _validate_canonical_polish_config(config: NewtonResidualPolishConfig) -> None:
    if type(config) is not NewtonResidualPolishConfig:
        raise ValueError("correction evidence requires the canonical NewtonResidualPolishConfig type")
    for name in ("max_iterations", "max_line_search_steps"):
        if type(getattr(config, name)) is not int:
            raise ValueError(f"residual-polish config {name} must be an integer")
    for name in ("gradient_absolute_tolerance", "gradient_relative_tolerance", "armijo", "backtrack"):
        if type(getattr(config, name)) is not float:
            raise ValueError(f"residual-polish config {name} must be a float")
    config.validate()


@dataclasses.dataclass(frozen=True, eq=False)
class CorrectionEndpoint:
    """One exact-work residual-Newton prefix and its independent metrics."""

    requested_corrections: int
    config: NewtonResidualPolishConfig
    scene_sha256: str
    objective_instance_sha256: str
    reference_position_sha256: str
    start: CorrectionStart
    initial_metrics: CommonStateMetrics
    positions: np.ndarray
    final_metrics: CommonStateMetrics
    solver_record: Mapping[str, object]
    _validation_scene: TetBenchmarkScene = dataclasses.field(repr=False, compare=False)
    _validation_problem: NewtonProblem = dataclasses.field(repr=False, compare=False)
    _validation_reference_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    solver_timing: Mapping[str, object] = dataclasses.field(repr=False, compare=False)
    endpoint_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.start) is not CorrectionStart:
            raise ValueError("correction endpoint requires the canonical CorrectionStart type")
        if type(self.requested_corrections) is not int or self.requested_corrections < 1:
            raise ValueError("requested_corrections must be a positive integer")
        _validate_canonical_polish_config(self.config)
        if self.config.max_iterations != self.requested_corrections:
            raise ValueError("residual-Newton max_iterations must equal the correction budget")
        if self.config.gradient_absolute_tolerance != 0.0 or self.config.gradient_relative_tolerance != 0.0:
            raise ValueError("exact correction prefixes require zero gradient tolerances")
        for name, value in (
            ("scene_sha256", self.scene_sha256),
            ("objective_instance_sha256", self.objective_instance_sha256),
            ("reference_position_sha256", self.reference_position_sha256),
        ):
            _sha256(value, name)
        if self.start.scene_sha256 != self.scene_sha256:
            raise ValueError("correction start belongs to a different scene")
        if self.start.objective_instance_sha256 != self.objective_instance_sha256:
            raise ValueError("correction start belongs to a different objective")
        if self.start.reference_position_sha256 != self.reference_position_sha256:
            raise ValueError("correction start uses a different accepted reference")

        positions, reference, actual_scene_sha256, actual_objective_sha256 = _validate_metrics_against_context(
            self._validation_scene,
            self._validation_problem,
            self.positions,
            self._validation_reference_positions,
            self.final_metrics,
            label="correction endpoint",
        )
        if self.scene_sha256 != actual_scene_sha256:
            raise ValueError("correction endpoint scene identity disagrees with its validation context")
        if self.objective_instance_sha256 != actual_objective_sha256:
            raise ValueError("correction endpoint objective identity disagrees with its validation context")
        if self.reference_position_sha256 != _array_digest(reference):
            raise ValueError("correction endpoint reference identity disagrees with its validation context")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "_validation_reference_positions", reference)
        solver_record = _freeze_json(self.solver_record)
        solver_timing = _freeze_json(self.solver_timing)
        object.__setattr__(self, "solver_record", solver_record)
        object.__setattr__(self, "solver_timing", solver_timing)

        canonical_problem, _, _ = _validated_canonical_problem(
            self._validation_scene,
            self._validation_problem,
        )

        if not _finite_metrics(self.initial_metrics) or not _finite_metrics(self.final_metrics):
            raise ValueError("correction endpoint metrics must be finite")
        if self.initial_metrics.position_sha256 != self.start.position_sha256:
            raise ValueError("initial metrics do not describe the correction start")
        if not _metrics_equal(self.initial_metrics, self.start.metrics):
            raise ValueError("independent initial metrics disagree with correction-start evidence")
        measured_initial = evaluate_common_state(
            canonical_problem,
            self.start.positions,
            reference_positions=reference,
        )
        if not _metrics_equal(self.initial_metrics, measured_initial):
            raise ValueError("initial metrics disagree with independent objective evaluation")
        if self.final_metrics.position_sha256 != _array_digest(positions):
            raise ValueError("final metrics do not describe the correction endpoint")

        _validate_solver_record(solver_record, self.config, self.requested_corrections)
        trace = solver_record["trace"]
        _validate_solver_timing(solver_timing, len(trace))
        first = trace[0]
        final = trace[-1]
        for recorded, measured, label in (
            (first.get("objective"), self.initial_metrics.objective, "initial objective"),
            (first.get("gradient_norm"), self.initial_metrics.gradient_norm, "initial gradient"),
            (first.get("relative_residual"), self.initial_metrics.relative_residual, "initial residual"),
            (final.get("objective"), self.final_metrics.objective, "final objective"),
            (final.get("gradient_norm"), self.final_metrics.gradient_norm, "final gradient"),
            (final.get("relative_residual"), self.final_metrics.relative_residual, "final residual"),
        ):
            if recorded != measured:
                raise ValueError(f"independent metrics disagree with the {label}")

        replay = solve_newton_residual_polish(canonical_problem, self.start.positions, self.config)
        if _array_digest(replay.x.detach().numpy()) != _array_digest(positions):
            raise ValueError("correction endpoint positions do not match deterministic solver replay")
        if _canonical_digest(replay.deterministic_record()) != _canonical_digest(_thaw_json(solver_record)):
            raise ValueError("residual-Newton record does not match deterministic solver replay")

        object.__setattr__(self, "endpoint_sha256", _canonical_digest(self._payload()))

    @property
    def applied_corrections(self) -> int:
        return int(self.solver_record["accepted_iterations"])

    @property
    def exact_budget_completed(self) -> bool:
        return (
            self.solver_record["reason"] in ("max_iterations", "gradient")
            and self.applied_corrections == self.requested_corrections
        )

    @property
    def saturated(self) -> bool:
        return (
            self.solver_record["reason"] == "gradient"
            and bool(self.solver_record["converged"])
            and self.applied_corrections < self.requested_corrections
        )

    @property
    def state_valid(self) -> bool:
        metrics = self.final_metrics
        return (
            _finite_metrics(metrics)
            and metrics.max_pin_error_m == 0.0
            and metrics.inverted_tet_fraction == 0.0
            and metrics.determinant_min > 0.0
            and metrics.minimum_singular_value > 0.0
        )

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _CONTRACT,
            "requested_corrections": self.requested_corrections,
            "applied_corrections": self.applied_corrections,
            "exact_budget_completed": self.exact_budget_completed,
            "saturated": self.saturated,
            "state_valid": self.state_valid,
            "config": self.config.deterministic_record(),
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "reference_position_sha256": self.reference_position_sha256,
            "start": self.start.as_dict(),
            "initial_metrics": self.initial_metrics.as_dict(),
            "position_sha256": _array_digest(self.positions),
            "final_metrics": self.final_metrics.as_dict(),
            "solver": _thaw_json(self.solver_record),
        }

    def as_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["endpoint_sha256"] = self.endpoint_sha256
        return payload

    def timing_record(self) -> dict[str, object]:
        payload = {
            "contract": "pss-dense-residual-correction-prefix-timing-v1",
            "performance_evidence": False,
            "measurement_provenance": _DIAGNOSTIC_TIMING_PROVENANCE,
            "endpoint_sha256": self.endpoint_sha256,
            "solver": _thaw_json(self.solver_timing),
        }
        payload["timing_sha256"] = _canonical_digest(payload)
        return payload


@dataclasses.dataclass(frozen=True, eq=False)
class CorrectionComparison:
    """Per-metric promotion gate against a fixed comparator state."""

    endpoint_sha256: str
    comparator_evidence_sha256: str
    scene_sha256: str
    objective_instance_sha256: str
    reference_position_sha256: str
    comparator_position_sha256: str
    residual_ratio: float
    free_rms_ratio: float
    objective_delta: float
    objective_roundoff_guard: float
    exact_budget_completed: bool
    state_valid: bool
    comparison_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.exact_budget_completed) is not bool or type(self.state_valid) is not bool:
            raise ValueError("correction comparison decision fields must be booleans")
        for name in ("residual_ratio", "free_rms_ratio", "objective_delta", "objective_roundoff_guard"):
            if type(getattr(self, name)) is not float:
                raise ValueError(f"correction comparison {name} must be a float")
        for name in (
            "endpoint_sha256",
            "comparator_evidence_sha256",
            "scene_sha256",
            "objective_instance_sha256",
            "reference_position_sha256",
            "comparator_position_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            not math.isfinite(self.residual_ratio)
            or self.residual_ratio < 0.0
            or not math.isfinite(self.free_rms_ratio)
            or self.free_rms_ratio < 0.0
            or not math.isfinite(self.objective_delta)
            or not math.isfinite(self.objective_roundoff_guard)
            or self.objective_roundoff_guard < 0.0
        ):
            raise ValueError("correction comparison scalars must be finite and valid")
        object.__setattr__(self, "comparison_sha256", _canonical_digest(self._payload()))

    @property
    def residual_no_worse(self) -> bool:
        return bool(self.residual_ratio <= 1.0)

    @property
    def free_rms_no_worse(self) -> bool:
        return bool(self.free_rms_ratio <= 1.0)

    @property
    def objective_no_worse(self) -> bool:
        return bool(self.objective_delta <= self.objective_roundoff_guard)

    @property
    def passed(self) -> bool:
        return bool(
            self.exact_budget_completed
            and self.state_valid
            and self.residual_no_worse
            and self.free_rms_no_worse
            and self.objective_no_worse
        )

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _COMPARISON_CONTRACT,
            "endpoint_sha256": self.endpoint_sha256,
            "comparator_evidence_sha256": self.comparator_evidence_sha256,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "reference_position_sha256": self.reference_position_sha256,
            "comparator_position_sha256": self.comparator_position_sha256,
            "residual_ratio": self.residual_ratio,
            "free_rms_ratio": self.free_rms_ratio,
            "objective_delta": self.objective_delta,
            "objective_roundoff_guard": self.objective_roundoff_guard,
            "residual_no_worse": self.residual_no_worse,
            "free_rms_no_worse": self.free_rms_no_worse,
            "objective_no_worse": self.objective_no_worse,
            "exact_budget_completed": self.exact_budget_completed,
            "state_valid": self.state_valid,
            "passed": self.passed,
        }

    def as_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["comparison_sha256"] = self.comparison_sha256
        return payload


def run_dense_residual_prefix(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    start: CorrectionStart,
    reference_positions: np.ndarray | torch.Tensor,
    correction_budget: int,
    *,
    base_config: NewtonResidualPolishConfig | None = None,
) -> CorrectionEndpoint:
    """Run an exact-work, residual-globalized dense Newton prefix.

    Args:
        scene: Exact benchmark scene shared by all compared methods.
        problem: Common stable-Neo-Hookean objective built from ``scene``.
        start: Content-addressed full-position initializer.
        reference_positions: Accepted reference positions [m], shape
            ``[V, 3]``.
        correction_budget: Exact number of requested Newton outer updates.
        base_config: Optional residual-polish globalization settings. Gradient
            tolerances and iteration count are replaced by the fixed-prefix
            contract.

    Returns:
        Immutable, content-addressed endpoint plus separate diagnostic timing.
    """
    if type(correction_budget) is not int:
        raise ValueError("correction_budget must be a positive integer")
    if type(start) is not CorrectionStart:
        raise ValueError("correction prefix requires the canonical CorrectionStart type")
    budget = int(correction_budget)
    if budget < 1:
        raise ValueError("correction_budget must be a positive integer")

    canonical_problem, scene_sha256, objective_instance_sha256 = _validated_canonical_problem(scene, problem)

    default_config = NewtonResidualPolishConfig(
        max_iterations=budget,
        gradient_absolute_tolerance=0.0,
        gradient_relative_tolerance=0.0,
        armijo=1.0e-4,
        backtrack=0.5,
        max_line_search_steps=30,
    )
    if base_config is not None:
        _validate_canonical_polish_config(base_config)
    source_config = default_config if base_config is None else base_config
    config = NewtonResidualPolishConfig(
        max_iterations=budget,
        gradient_absolute_tolerance=0.0,
        gradient_relative_tolerance=0.0,
        armijo=source_config.armijo,
        backtrack=source_config.backtrack,
        max_line_search_steps=source_config.max_line_search_steps,
    )
    _validate_canonical_polish_config(config)

    reference = _readonly_positions(reference_positions, "reference positions")
    if reference.shape != start.positions.shape:
        raise ValueError("reference positions do not match the correction start")
    reference_position_sha256 = _array_digest(reference)
    if start.scene_sha256 != scene_sha256:
        raise ValueError("correction start belongs to a different scene")
    if start.objective_instance_sha256 != objective_instance_sha256:
        raise ValueError("correction start belongs to a different objective")
    if start.reference_position_sha256 != reference_position_sha256:
        raise ValueError("correction start uses a different accepted reference")
    initial_metrics = evaluate_common_state(canonical_problem, start.positions, reference_positions=reference)
    if not _metrics_equal(initial_metrics, start.metrics):
        raise ValueError("correction start metrics changed under independent evaluation")
    result = solve_newton_residual_polish(canonical_problem, start.positions, config)
    positions = _readonly_positions(result.x, "correction endpoint positions")
    final_metrics = evaluate_common_state(canonical_problem, positions, reference_positions=reference)
    return CorrectionEndpoint(
        requested_corrections=budget,
        config=config,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        reference_position_sha256=reference_position_sha256,
        start=start,
        initial_metrics=initial_metrics,
        positions=positions,
        final_metrics=final_metrics,
        solver_record=result.deterministic_record(),
        solver_timing=result.timing_record(),
        _validation_scene=scene,
        _validation_problem=canonical_problem,
        _validation_reference_positions=reference,
    )


def compare_endpoint_to_state(
    endpoint: CorrectionEndpoint,
    comparator: CorrectionStart,
) -> CorrectionComparison:
    """Compare one exact prefix to VBD-quality residual, error, and energy."""
    if type(endpoint) is not CorrectionEndpoint or type(comparator) is not CorrectionStart:
        raise ValueError("correction comparison requires canonical endpoint and comparator types")
    if comparator.scene_sha256 != endpoint.scene_sha256:
        raise ValueError("comparator belongs to a different scene")
    if comparator.objective_instance_sha256 != endpoint.objective_instance_sha256:
        raise ValueError("comparator belongs to a different objective")
    if comparator.reference_position_sha256 != endpoint.reference_position_sha256:
        raise ValueError("comparator uses a different accepted reference")
    metrics = comparator.metrics
    if not _finite_metrics(metrics):
        raise ValueError("comparator metrics must be finite")
    candidate_rms = endpoint.final_metrics.free_rms_error_m
    comparator_rms = metrics.free_rms_error_m
    if candidate_rms is None or comparator_rms is None:
        raise ValueError("both states must have reference free-RMS metrics")
    residual_denominator = max(metrics.relative_residual, np.finfo(np.float64).tiny)
    rms_denominator = max(comparator_rms, np.finfo(np.float64).tiny)
    objective_guard = float(
        _OBJECTIVE_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * max(1.0, abs(endpoint.final_metrics.objective), abs(metrics.objective))
    )
    return CorrectionComparison(
        endpoint_sha256=endpoint.endpoint_sha256,
        comparator_evidence_sha256=comparator.evidence_sha256,
        scene_sha256=endpoint.scene_sha256,
        objective_instance_sha256=endpoint.objective_instance_sha256,
        reference_position_sha256=endpoint.reference_position_sha256,
        comparator_position_sha256=comparator.position_sha256,
        residual_ratio=float(endpoint.final_metrics.relative_residual / residual_denominator),
        free_rms_ratio=float(candidate_rms / rms_denominator),
        objective_delta=float(endpoint.final_metrics.objective - metrics.objective),
        objective_roundoff_guard=objective_guard,
        exact_budget_completed=endpoint.exact_budget_completed,
        state_valid=endpoint.state_valid,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class CorrectionLadder:
    """One initializer's independently restarted correction-depth ladder."""

    start: CorrectionStart
    comparator: CorrectionStart
    endpoints: tuple[CorrectionEndpoint, ...]
    comparisons: tuple[CorrectionComparison, ...]
    ladder_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.start) is not CorrectionStart or type(self.comparator) is not CorrectionStart:
            raise ValueError("correction ladder requires canonical start and comparator types")
        endpoints = tuple(self.endpoints)
        comparisons = tuple(self.comparisons)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "comparisons", comparisons)
        if not endpoints or len(endpoints) != len(comparisons):
            raise ValueError("correction ladder endpoints and comparisons must have the same nonzero length")
        if any(type(endpoint) is not CorrectionEndpoint for endpoint in endpoints) or any(
            type(comparison) is not CorrectionComparison for comparison in comparisons
        ):
            raise ValueError("correction ladder requires canonical endpoint and comparison types")
        budgets = [endpoint.requested_corrections for endpoint in endpoints]
        if budgets != sorted(set(budgets)):
            raise ValueError("correction budgets must be strictly increasing and unique")
        for endpoint, comparison in zip(endpoints, comparisons, strict=True):
            if endpoint.start.evidence_sha256 != self.start.evidence_sha256:
                raise ValueError("correction ladder endpoint uses the wrong start")
            if comparison.endpoint_sha256 != endpoint.endpoint_sha256:
                raise ValueError("correction ladder comparison uses the wrong endpoint")
            if comparison.comparator_evidence_sha256 != self.comparator.evidence_sha256:
                raise ValueError("correction ladder comparison uses the wrong comparator")
            if comparison.exact_budget_completed != endpoint.exact_budget_completed:
                raise ValueError("correction comparison relabels exact-budget completion")
            if comparison.state_valid != endpoint.state_valid:
                raise ValueError("correction comparison relabels endpoint validity")
            expected_comparison = compare_endpoint_to_state(endpoint, self.comparator)
            if _canonical_digest(comparison.as_dict()) != _canonical_digest(expected_comparison.as_dict()):
                raise ValueError("correction comparison does not match independent metrics")
        object.__setattr__(self, "ladder_sha256", _canonical_digest(self._payload()))

    @property
    def smallest_passing_budget(self) -> int | None:
        return smallest_passing_budget(self.endpoints, self.comparisons, self.comparator)

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _LADDER_CONTRACT,
            "start_evidence_sha256": self.start.evidence_sha256,
            "comparator_evidence_sha256": self.comparator.evidence_sha256,
            "endpoints": [endpoint.as_dict() for endpoint in self.endpoints],
            "comparisons": [comparison.as_dict() for comparison in self.comparisons],
            "smallest_passing_budget": self.smallest_passing_budget,
        }

    def as_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["ladder_sha256"] = self.ladder_sha256
        return payload


@dataclasses.dataclass(frozen=True, eq=False)
class TransitionCorrectionCeiling:
    """One authenticated PR transition's dense correction ceiling."""

    history_manifest_sha256: str
    history_static_sha256: str
    history_chain_sha256: str
    transition_sha256: str
    coordinate: Mapping[str, object]
    scene_sha256: str
    objective_instance_sha256: str
    reference_position_sha256: str
    reference_record_sha256: str
    reference_metrics: CommonStateMetrics
    vbd_k1: CorrectionStart
    vbd_k4: CorrectionStart
    ladders: tuple[CorrectionLadder, ...]
    _validation_history: PRSceneHistory = dataclasses.field(repr=False, compare=False)
    _validation_chain: PRHistoryChain = dataclasses.field(repr=False, compare=False)
    _validation_transition: HistoryTransition = dataclasses.field(repr=False, compare=False)
    _validation_expected_history_chain_sha256: str = dataclasses.field(repr=False, compare=False)
    _validation_scene: TetBenchmarkScene = dataclasses.field(repr=False, compare=False)
    _validation_problem: NewtonProblem = dataclasses.field(repr=False, compare=False)
    _validation_reference_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    timings: Mapping[str, object] = dataclasses.field(repr=False, compare=False)
    result_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        coordinate_snapshot = _thaw_json(self.coordinate)
        if not isinstance(coordinate_snapshot, Mapping):
            raise ValueError("transition ceiling coordinate must be a mapping")
        history = _reconstruct_canonical_history(self._validation_history)
        source_chain = self._validation_chain
        ordinal = _chain_member_ordinal(source_chain, self._validation_transition)
        chain = _snapshot_chain(source_chain)
        transition = chain.transitions[ordinal]
        if type(self.vbd_k1) is not CorrectionStart or type(self.vbd_k4) is not CorrectionStart:
            raise ValueError("transition ceiling requires canonical VBD evidence types")
        expected_chain_sha256 = _sha256(
            self._validation_expected_history_chain_sha256,
            "expected_history_chain_sha256",
        )
        if type(self._validation_scene) is not TetBenchmarkScene or any(
            callable(value) for value in vars(self._validation_scene).values()
        ):
            raise ValueError("transition ceiling requires the canonical TetBenchmarkScene implementation")
        _verify_history_chain_raw_content(history, chain)
        PRHistoryChain.verify(chain)
        if chain.chain_sha256 != expected_chain_sha256:
            raise ValueError("transition ceiling chain does not match its external SHA-256 pin")
        if chain.manifest.as_dict() != history.manifest.as_dict():
            raise ValueError("transition ceiling chain belongs to a different canonical history")
        if (
            chain.initial_checkpoint.as_dict() != history.initial_checkpoint.as_dict()
            or chain.prior_chain_sha256 is not None
        ):
            raise ValueError("transition ceiling requires a canonical root history chain")
        matching = [item for item in chain.transitions if item.coordinate == transition.coordinate]
        if len(matching) != 1 or matching[0].as_dict() != transition.as_dict():
            raise ValueError("transition ceiling record is not an exact chain member")
        _verify_transition_by_canonical_replay(history, chain, transition)
        canonical_applied = PRSceneHistory.apply_callback(history, transition.input_state)
        if canonical_applied.as_dict() != transition.applied_state.as_dict():
            raise ValueError("transition ceiling applied state changed from the canonical callback")
        canonical_scene = PRSceneHistory.build_atomic_scene(history, transition.input_state, transition.applied_state)
        if canonical_scene.manifest() != self._validation_scene.manifest():
            raise ValueError("transition ceiling scene changed from canonical reconstruction")
        canonical_problem = build_common_problem(canonical_scene)
        _, canonical_objective_sha256 = _validated_problem_identity(canonical_scene, canonical_problem)
        validation_reference = _readonly_positions(
            self._validation_reference_positions,
            "transition ceiling validation reference",
        )
        if not np.array_equal(validation_reference, transition.reference_positions):
            raise ValueError("transition ceiling reference changed from its chain transition")
        canonical_reference_metrics = evaluate_common_state(
            canonical_problem,
            validation_reference,
            reference_positions=validation_reference,
        )
        canonical_reference_record_sha256 = _validate_accepted_reference_record(
            transition,
            canonical_problem,
            canonical_reference_metrics,
        )
        expected_identities = {
            "history_manifest_sha256": history.manifest.manifest_sha256,
            "history_static_sha256": history.static_bundle.static_sha256,
            "history_chain_sha256": chain.chain_sha256,
            "transition_sha256": transition.transition_sha256,
            "scene_sha256": str(canonical_scene.manifest()["scene_sha256"]),
            "objective_instance_sha256": canonical_objective_sha256,
            "reference_position_sha256": _array_digest(validation_reference),
            "reference_record_sha256": canonical_reference_record_sha256,
        }
        for name, expected in expected_identities.items():
            if getattr(self, name) != expected:
                raise ValueError(f"transition ceiling {name} changed from canonical history evidence")
        if _canonical_digest(coordinate_snapshot) != _canonical_digest(transition.coordinate.as_dict()):
            raise ValueError("transition ceiling coordinate changed from canonical history evidence")
        if not _metrics_equal(self.reference_metrics, canonical_reference_metrics):
            raise ValueError("transition ceiling reference metrics changed from canonical history evidence")
        for name in (
            "history_manifest_sha256",
            "history_static_sha256",
            "history_chain_sha256",
            "transition_sha256",
            "scene_sha256",
            "objective_instance_sha256",
            "reference_position_sha256",
            "reference_record_sha256",
        ):
            _sha256(getattr(self, name), name)
        coordinate = _freeze_json(coordinate_snapshot)
        timings = _freeze_json(self.timings)
        ladders = tuple(self.ladders)
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "timings", timings)
        object.__setattr__(self, "ladders", ladders)
        reference, validated_reference, actual_scene_sha256, actual_objective_sha256 = (
            _validate_metrics_against_context(
                self._validation_scene,
                self._validation_problem,
                self._validation_reference_positions,
                self._validation_reference_positions,
                self.reference_metrics,
                label="accepted reference",
            )
        )
        if self.scene_sha256 != actual_scene_sha256:
            raise ValueError("transition scene identity disagrees with its validation context")
        if self.objective_instance_sha256 != actual_objective_sha256:
            raise ValueError("transition objective identity disagrees with its validation context")
        if self.reference_position_sha256 != _array_digest(reference):
            raise ValueError("transition reference identity disagrees with its validation context")
        if not np.array_equal(reference, validated_reference):
            raise ValueError("transition reference validation context changed")
        object.__setattr__(self, "_validation_reference_positions", reference)
        if not ladders:
            raise ValueError("transition correction ceiling requires at least one start ladder")
        if any(type(ladder) is not CorrectionLadder for ladder in ladders):
            raise ValueError("transition ceiling requires canonical correction ladder types")
        if self.reference_metrics.position_sha256 != self.reference_position_sha256:
            raise ValueError("reference metrics do not describe the accepted reference")
        if not _finite_metrics(self.reference_metrics):
            raise ValueError("accepted-reference metrics must be finite")
        if (
            self.reference_metrics.max_pin_error_m != 0.0
            or self.reference_metrics.inverted_tet_fraction != 0.0
            or self.reference_metrics.determinant_min <= 0.0
            or self.reference_metrics.minimum_singular_value <= 0.0
        ):
            raise ValueError("accepted reference is not a valid common-objective state")
        for evidence, role in ((self.vbd_k1, "vbd-k1"), (self.vbd_k4, "vbd-k4")):
            if evidence.role != role:
                raise ValueError(f"transition ceiling requires fresh {role} evidence")
            if (
                evidence.scene_sha256 != self.scene_sha256
                or evidence.objective_instance_sha256 != self.objective_instance_sha256
                or evidence.reference_position_sha256 != self.reference_position_sha256
            ):
                raise ValueError(f"{role} evidence belongs to another transition objective")
            _validate_vbd_state_evidence(
                evidence,
                role=role,
                scene=self._validation_scene,
                problem=self._validation_problem,
            )
        if self.ladders[0].start.evidence_sha256 != self.vbd_k1.evidence_sha256:
            raise ValueError("the primary correction ladder must start from fresh VBD K1")
        if any(ladder.comparator.evidence_sha256 != self.vbd_k4.evidence_sha256 for ladder in ladders):
            raise ValueError("every correction ladder must use fresh VBD K4 as comparator")
        for name, evidence in (("vbd_k1", self.vbd_k1), ("vbd_k4", self.vbd_k4)):
            timing = timings.get(name) if isinstance(timings, Mapping) else None
            if not isinstance(timing, Mapping):
                raise ValueError(f"transition ceiling is missing {name} timing evidence")
            _validate_vbd_timing_record(timing, evidence)
        timing_ladders = timings.get("ladders") if isinstance(timings, Mapping) else None
        if not isinstance(timing_ladders, tuple) or len(timing_ladders) != len(ladders):
            raise ValueError("transition ceiling ladder timings are incomplete")
        for ladder, timing_ladder in zip(ladders, timing_ladders, strict=True):
            if not isinstance(timing_ladder, Mapping):
                raise ValueError("transition ceiling ladder timing entry is invalid")
            if timing_ladder.get("ladder_sha256") != ladder.ladder_sha256:
                raise ValueError("transition ceiling timing uses the wrong correction ladder")
            if timing_ladder.get("start_evidence_sha256") != ladder.start.evidence_sha256:
                raise ValueError("transition ceiling timing uses the wrong correction start")
            endpoint_timings = timing_ladder.get("endpoints")
            if not isinstance(endpoint_timings, tuple) or len(endpoint_timings) != len(ladder.endpoints):
                raise ValueError("transition ceiling endpoint timings are incomplete")
            for endpoint, timing in zip(ladder.endpoints, endpoint_timings, strict=True):
                if _thaw_json(timing) != endpoint.timing_record():
                    raise ValueError("transition ceiling endpoint timing changed")
        object.__setattr__(self, "result_sha256", _canonical_digest(self._payload()))

    @property
    def primary_smallest_passing_budget(self) -> int | None:
        return self.ladders[0].smallest_passing_budget

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _TRANSITION_CONTRACT,
            "history_manifest_sha256": self.history_manifest_sha256,
            "history_static_sha256": self.history_static_sha256,
            "history_chain_sha256": self.history_chain_sha256,
            "transition_sha256": self.transition_sha256,
            "coordinate": _thaw_json(self.coordinate),
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "reference_position_sha256": self.reference_position_sha256,
            "reference_record_sha256": self.reference_record_sha256,
            "reference_metrics": self.reference_metrics.as_dict(),
            "vbd_k1": self.vbd_k1.as_dict(),
            "vbd_k4": self.vbd_k4.as_dict(),
            "ladders": [ladder.as_dict() for ladder in self.ladders],
            "primary_smallest_passing_budget": self.primary_smallest_passing_budget,
        }

    def as_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["result_sha256"] = self.result_sha256
        return payload

    def timing_record(self) -> dict[str, object]:
        payload = {
            "contract": "pss-pr-transition-correction-ceiling-timing-v1",
            "performance_evidence": False,
            "measurement_provenance": _DIAGNOSTIC_TIMING_PROVENANCE,
            "result_sha256": self.result_sha256,
            "values": _thaw_json(self.timings),
        }
        payload["timing_sha256"] = _canonical_digest(payload)
        return payload


def smallest_passing_budget(
    endpoints: Sequence[CorrectionEndpoint],
    comparisons: Sequence[CorrectionComparison],
    comparator: CorrectionStart,
) -> int | None:
    """Return the smallest exact correction budget passing every metric."""
    if type(comparator) is not CorrectionStart:
        raise ValueError("correction-budget selection requires the canonical comparator type")
    endpoints = tuple(endpoints)
    comparisons = tuple(comparisons)
    if len(endpoints) != len(comparisons) or not endpoints:
        raise ValueError("endpoint and comparison ladders must have the same nonzero length")
    if any(type(endpoint) is not CorrectionEndpoint for endpoint in endpoints) or any(
        type(comparison) is not CorrectionComparison for comparison in comparisons
    ):
        raise ValueError("correction-budget selection requires canonical endpoint and comparison types")
    budgets = [endpoint.requested_corrections for endpoint in endpoints]
    if budgets != sorted(set(budgets)):
        raise ValueError("correction budgets must be strictly increasing and unique")
    start_evidence_sha256 = endpoints[0].start.evidence_sha256
    passing_budget = None
    for endpoint, comparison in zip(endpoints, comparisons, strict=True):
        if endpoint.start.evidence_sha256 != start_evidence_sha256:
            raise ValueError("correction endpoints do not share one initializer")
        if comparison.endpoint_sha256 != endpoint.endpoint_sha256:
            raise ValueError("correction comparison does not match its endpoint")
        if comparison.exact_budget_completed != endpoint.exact_budget_completed:
            raise ValueError("correction comparison relabels exact-budget completion")
        if comparison.state_valid != endpoint.state_valid:
            raise ValueError("correction comparison relabels endpoint validity")
        if _canonical_digest(comparison.as_dict()) != _canonical_digest(
            compare_endpoint_to_state(endpoint, comparator).as_dict()
        ):
            raise ValueError("correction comparison does not match independent metrics")
        if passing_budget is None and endpoint.exact_budget_completed and comparison.passed:
            passing_budget = endpoint.requested_corrections
    return passing_budget


def _validate_accepted_reference_record(
    transition: HistoryTransition,
    problem: NewtonProblem,
    metrics: CommonStateMetrics,
) -> str:
    record = _thaw_json(_freeze_json(transition.reference_record))
    method = record.get("method")
    if method not in _REFERENCE_METHOD_POLICIES:
        raise ValueError("accepted-reference record changed method")
    policy = _REFERENCE_METHOD_POLICIES[method]
    if policy is not None:
        policy_name, policy_value = policy
        if record.get(policy_name) != policy_value:
            raise ValueError(f"accepted-reference record changed {policy_name}")
    required_identities = {
        "scene_sha256": transition.scene_sha256,
        "objective_instance_sha256": transition.objective_instance_sha256,
        "position_sha256": metrics.position_sha256,
        "accepted": True,
    }
    for name, expected in required_identities.items():
        if record.get(name) != expected:
            raise ValueError(f"accepted-reference record changed {name}")
    if _thaw_json(record.get("failures")) != []:
        raise ValueError("accepted-reference record must have no failures")
    for name, expected in (
        ("final_objective", metrics.objective),
        ("final_gradient_norm", metrics.gradient_norm),
        ("final_relative_residual", metrics.relative_residual),
    ):
        if record.get(name) != expected:
            raise ValueError(f"accepted-reference record changed {name}")
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("accepted-reference record is missing its Newton config")
    native_config = NewtonConfig(
        max_iterations=50,
        gradient_absolute_tolerance=1.0e-10,
        gradient_relative_tolerance=1.0e-10,
        step_relative_tolerance=1.0e-14,
    )
    retry_config = dataclasses.replace(native_config, step_relative_tolerance=0.0)
    allowed_configs = [dataclasses.asdict(retry_config)]
    if method == "dense-cpu-newton-float64":
        allowed_configs.append(dataclasses.asdict(native_config))
    if dict(_thaw_json(config)) not in allowed_configs:
        raise ValueError("accepted-reference Newton config changed from the canonical history policy")
    gradient_limit = max(
        native_config.gradient_absolute_tolerance,
        native_config.gradient_relative_tolerance * problem.residual_scale,
    )
    if metrics.gradient_norm > gradient_limit:
        raise ValueError("accepted reference exceeds its recorded gradient gate")
    verification_displacement = record.get("verification_displacement_relative")
    alternate_displacement = record.get("alternate_start_displacement_relative")
    if (
        not isinstance(verification_displacement, numbers.Real)
        or isinstance(verification_displacement, bool)
        or not math.isfinite(float(verification_displacement))
        or not 0.0 <= float(verification_displacement) <= 1.0e-12
    ):
        raise ValueError("accepted reference exceeds its verification-displacement gate")
    if (
        not isinstance(alternate_displacement, numbers.Real)
        or isinstance(alternate_displacement, bool)
        or not math.isfinite(float(alternate_displacement))
        or not 0.0 <= float(alternate_displacement) <= 1.0e-9
    ):
        raise ValueError("accepted reference exceeds its alternate-displacement gate")
    if record.get("verification_converged") is not True or record.get("verification_reason") != "gradient":
        raise ValueError("accepted reference lacks converged independent verification")
    return _canonical_digest(_thaw_json(record))


def evaluate_pr_transition_correction_ceiling(
    history: PRSceneHistory,
    chain: PRHistoryChain,
    transition: HistoryTransition,
    *,
    expected_history_chain_sha256: str,
    correction_budgets: Sequence[int] = (1, 2),
    optional_starts: Sequence[CorrectionStart] = (),
    vbd_device: str = "cpu",
    vbd_warmup: bool = False,
    vbd_repeats: int = 1,
) -> TransitionCorrectionCeiling:
    """Evaluate fresh VBD K1 plus dense corrections against fresh VBD K4.

    Args:
        history: Exact PR callback history that owns ``transition``.
        chain: Verified accepted chain containing ``transition``.
        transition: Accepted atomic transition with a float64 dense reference.
        expected_history_chain_sha256: Externally pinned chain identity.
        correction_budgets: Strictly increasing independent correction depths.
        optional_starts: Additional pre-evaluated starts on the same objective.
        vbd_device: Must be ``"cpu"`` so VBD quality evidence can be replayed exactly.
        vbd_warmup: Whether each VBD run performs one untimed warmup.
        vbd_repeats: Number of fresh-state VBD timing repeats.

    Returns:
        Content-addressed quality evidence with diagnostic timings separated.
    """
    history = _reconstruct_canonical_history(history)
    ordinal = _chain_member_ordinal(chain, transition)
    chain = _snapshot_chain(chain)
    transition = chain.transitions[ordinal]
    budgets = tuple(correction_budgets)
    if (
        not budgets
        or any(type(value) is not int or value < 1 for value in budgets)
        or list(budgets) != sorted(set(budgets))
    ):
        raise ValueError("correction budgets must be strictly increasing positive integers")
    budgets = tuple(int(value) for value in budgets)
    if vbd_device != "cpu":
        raise ValueError("correction-ceiling VBD evidence requires deterministic scalar CPU execution")
    _sha256(expected_history_chain_sha256, "expected_history_chain_sha256")
    _verify_history_chain_raw_content(history, chain)
    PRHistoryChain.verify(chain)
    if chain.chain_sha256 != expected_history_chain_sha256:
        raise ValueError("PR history chain does not match the externally pinned SHA-256")
    if chain.manifest.as_dict() != history.manifest.as_dict():
        raise ValueError("PR history chain belongs to a different history object")
    if (
        chain.initial_checkpoint.as_dict() != history.initial_checkpoint.as_dict()
        or chain.prior_chain_sha256 is not None
    ):
        raise ValueError("correction evidence requires the canonical root PR history chain")
    matching_transitions = [item for item in chain.transitions if item.coordinate == transition.coordinate]
    if len(matching_transitions) != 1 or matching_transitions[0].as_dict() != transition.as_dict():
        raise ValueError("transition is not an exact member of the verified PR history chain")
    if transition.manifest_sha256 != history.manifest.manifest_sha256:
        raise ValueError("transition belongs to a different PR history")
    if transition.topology_sha256 != history.static_bundle.topology_sha256:
        raise ValueError("transition topology does not match the PR history")
    if transition.material_sha256 != history.static_bundle.material_sha256:
        raise ValueError("transition materials do not match the PR history")
    if transition.reference_record.get("accepted") is not True:
        raise ValueError("transition does not contain an accepted dense reference")
    canonical_applied = PRSceneHistory.apply_callback(history, transition.input_state)
    if canonical_applied.as_dict() != transition.applied_state.as_dict():
        raise ValueError("transition applied state does not match the canonical PR callback")

    scene = PRSceneHistory.build_atomic_scene(history, transition.input_state, transition.applied_state)
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    if scene_sha256 != transition.scene_sha256:
        raise ValueError("reconstructed transition scene SHA-256 changed")
    problem = build_common_problem(scene)
    _, objective_instance_sha256 = _validated_problem_identity(scene, problem)
    if objective_instance_sha256 != transition.objective_instance_sha256:
        raise ValueError("reconstructed transition objective SHA-256 changed")
    inertial_target = problem.inertial_target.detach().numpy()
    if not np.array_equal(inertial_target, transition.inertial_target.astype(np.float64)):
        raise ValueError("reconstructed transition inertial target changed")

    reference = _readonly_positions(transition.reference_positions, "transition reference positions")
    reference_position_sha256 = _array_digest(reference)
    if transition.reference_record.get("position_sha256") != reference_position_sha256:
        raise ValueError("accepted-reference record does not bind the transition positions")
    reference_metrics = evaluate_common_state(problem, reference, reference_positions=reference)
    reference_record_sha256 = _validate_accepted_reference_record(transition, problem, reference_metrics)

    # These are deliberately separate calls: K4 is a fresh fixed-objective
    # comparator, never a continuation of the K1 state.
    vbd_k1_run = run_vbd(
        scene,
        1,
        device=vbd_device,
        warmup=vbd_warmup,
        repeats=vbd_repeats,
    )
    vbd_k4_run = run_vbd(
        scene,
        4,
        device=vbd_device,
        warmup=vbd_warmup,
        repeats=vbd_repeats,
    )
    vbd_k1 = build_vbd_correction_start(scene, problem, vbd_k1_run, reference)
    vbd_k4 = build_vbd_correction_start(scene, problem, vbd_k4_run, reference)

    starts = (vbd_k1, *tuple(optional_starts))
    ladders: list[CorrectionLadder] = []
    ladder_timings: list[dict[str, object]] = []
    for start in starts:
        if (
            start.scene_sha256 != scene_sha256
            or start.objective_instance_sha256 != objective_instance_sha256
            or start.reference_position_sha256 != reference_position_sha256
        ):
            raise ValueError("optional correction start belongs to another transition objective")
        independently_measured = evaluate_common_state(problem, start.positions, reference_positions=reference)
        if not _metrics_equal(independently_measured, start.metrics):
            raise ValueError("optional correction-start metrics changed under independent evaluation")
        endpoints = tuple(run_dense_residual_prefix(scene, problem, start, reference, budget) for budget in budgets)
        comparisons = tuple(compare_endpoint_to_state(endpoint, vbd_k4) for endpoint in endpoints)
        ladder = CorrectionLadder(start=start, comparator=vbd_k4, endpoints=endpoints, comparisons=comparisons)
        ladders.append(ladder)
        ladder_timings.append(
            {
                "ladder_sha256": ladder.ladder_sha256,
                "start_evidence_sha256": start.evidence_sha256,
                "endpoints": [endpoint.timing_record() for endpoint in endpoints],
            }
        )

    timings = {
        "vbd_k1": _vbd_timing_record(vbd_k1_run, vbd_k1),
        "vbd_k4": _vbd_timing_record(vbd_k4_run, vbd_k4),
        "ladders": ladder_timings,
    }
    return TransitionCorrectionCeiling(
        history_manifest_sha256=history.manifest.manifest_sha256,
        history_static_sha256=history.static_bundle.static_sha256,
        history_chain_sha256=chain.chain_sha256,
        transition_sha256=transition.transition_sha256,
        coordinate=transition.coordinate.as_dict(),
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        reference_position_sha256=reference_position_sha256,
        reference_record_sha256=reference_record_sha256,
        reference_metrics=reference_metrics,
        vbd_k1=vbd_k1,
        vbd_k4=vbd_k4,
        ladders=tuple(ladders),
        _validation_history=history,
        _validation_chain=chain,
        _validation_transition=transition,
        _validation_expected_history_chain_sha256=expected_history_chain_sha256,
        timings=timings,
        _validation_scene=scene,
        _validation_problem=problem,
        _validation_reference_positions=reference,
    )


def verify_correction_endpoint_record(
    record: Mapping[str, object],
    *,
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    reference_positions: np.ndarray | torch.Tensor,
    start_positions: np.ndarray | torch.Tensor,
    endpoint_positions: np.ndarray | torch.Tensor,
    start_vbd_result: VBDRunResult | None = None,
) -> None:
    """Verify endpoint content, raw states, objective, and redundant semantics."""
    if not isinstance(record, Mapping):
        raise ValueError("correction endpoint record must be a mapping")
    record = _thaw_json(_freeze_json(record))
    if not isinstance(record, Mapping):
        raise ValueError("correction endpoint record must canonicalize to a mapping")
    expected_fields = {
        "contract",
        "requested_corrections",
        "applied_corrections",
        "exact_budget_completed",
        "saturated",
        "state_valid",
        "config",
        "scene_sha256",
        "objective_instance_sha256",
        "reference_position_sha256",
        "start",
        "initial_metrics",
        "position_sha256",
        "final_metrics",
        "solver",
        "endpoint_sha256",
    }
    if set(record) != expected_fields:
        raise ValueError("correction endpoint record fields changed")
    if record.get("contract") != _CONTRACT:
        raise ValueError("correction endpoint contract changed")
    supplied = record.get("endpoint_sha256")
    _sha256(supplied, "endpoint_sha256")
    payload = dict(record)
    payload.pop("endpoint_sha256", None)
    if supplied != _canonical_digest(payload):
        raise ValueError("correction endpoint SHA-256 verification failed")

    requested = record.get("requested_corrections")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("serialized correction budget is invalid")
    config_record = record.get("config")
    if not isinstance(config_record, Mapping):
        raise ValueError("serialized correction config is missing")
    config_fields = {field.name for field in dataclasses.fields(NewtonResidualPolishConfig)}
    if not config_fields <= config_record.keys():
        raise ValueError("serialized correction config fields are incomplete")
    for name in ("max_iterations", "max_line_search_steps"):
        if type(config_record.get(name)) is not int:
            raise ValueError(f"serialized correction config {name} must be an integer")
    for name in ("gradient_absolute_tolerance", "gradient_relative_tolerance", "armijo", "backtrack"):
        if type(config_record.get(name)) is not float:
            raise ValueError(f"serialized correction config {name} must be a float")
    config = NewtonResidualPolishConfig(**{name: config_record[name] for name in config_fields})
    config.validate()
    expected_config_record = config.deterministic_record()
    if set(config_record) != set(expected_config_record) or _canonical_digest(dict(config_record)) != _canonical_digest(
        expected_config_record
    ):
        raise ValueError("serialized correction config contract changed")
    if config.max_iterations != requested:
        raise ValueError("serialized correction budget and config disagree")
    if config.gradient_absolute_tolerance != 0.0 or config.gradient_relative_tolerance != 0.0:
        raise ValueError("serialized correction prefix is not zero-tolerance")

    def parse_metrics(value: object, name: str) -> CommonStateMetrics:
        if not isinstance(value, Mapping):
            raise ValueError(f"serialized {name} metrics are missing")
        field_names = {field.name for field in dataclasses.fields(CommonStateMetrics)}
        if set(value) != field_names:
            raise ValueError(f"serialized {name} metrics fields changed")
        metrics = CommonStateMetrics(**{field_name: value[field_name] for field_name in field_names})
        _sha256(metrics.position_sha256, f"serialized {name} position_sha256")
        if not _finite_metrics(metrics):
            raise ValueError(f"serialized {name} metrics must be finite and physically structured")
        return metrics

    start = record.get("start")
    if not isinstance(start, Mapping) or start.get("contract") != _STATE_CONTRACT:
        raise ValueError("serialized correction-start evidence is missing")
    start_payload = dict(start)
    start_evidence_sha256 = start_payload.pop("evidence_sha256", None)
    _sha256(start_evidence_sha256, "serialized correction-start evidence_sha256")
    if start_evidence_sha256 != _canonical_digest(start_payload):
        raise ValueError("serialized correction-start evidence SHA-256 verification failed")
    start_metrics = parse_metrics(start.get("metrics"), "start")
    initial_metrics = parse_metrics(record.get("initial_metrics"), "initial")
    final_metrics = parse_metrics(record.get("final_metrics"), "final")
    if start_metrics.as_dict() != initial_metrics.as_dict():
        raise ValueError("serialized correction-start and initial metrics disagree")
    if start_metrics.position_sha256 != start.get("position_sha256"):
        raise ValueError("serialized correction-start position hash is inconsistent")
    if final_metrics.position_sha256 != record.get("position_sha256"):
        raise ValueError("serialized correction endpoint position hash is inconsistent")
    _sha256(start.get("position_sha256"), "serialized correction-start position_sha256")
    _sha256(record.get("position_sha256"), "serialized correction endpoint position_sha256")
    for name in ("scene_sha256", "objective_instance_sha256", "reference_position_sha256"):
        _sha256(record.get(name), name)
        if start.get(name) != record.get(name):
            raise ValueError(f"serialized correction start and endpoint disagree on {name}")

    start_array, reference, scene_sha256, objective_instance_sha256 = _validate_metrics_against_context(
        scene,
        problem,
        start_positions,
        reference_positions,
        start_metrics,
        label="serialized correction start",
    )
    endpoint_array, endpoint_reference, endpoint_scene_sha256, endpoint_objective_sha256 = (
        _validate_metrics_against_context(
            scene,
            problem,
            endpoint_positions,
            reference_positions,
            final_metrics,
            label="serialized correction endpoint",
        )
    )
    if scene_sha256 != record.get("scene_sha256") or endpoint_scene_sha256 != scene_sha256:
        raise ValueError("serialized correction scene identity changed")
    if (
        objective_instance_sha256 != record.get("objective_instance_sha256")
        or endpoint_objective_sha256 != objective_instance_sha256
    ):
        raise ValueError("serialized correction objective identity changed")
    if _array_digest(reference) != record.get("reference_position_sha256") or not np.array_equal(
        reference,
        endpoint_reference,
    ):
        raise ValueError("serialized correction reference identity changed")
    if _array_digest(start_array) != start.get("position_sha256"):
        raise ValueError("serialized correction-start raw positions changed")
    if _array_digest(endpoint_array) != record.get("position_sha256"):
        raise ValueError("serialized correction endpoint raw positions changed")

    start_role = start.get("role")
    if not isinstance(start_role, str) or not start_role:
        raise ValueError("serialized correction-start role is invalid")
    start_provenance = start.get("provenance")
    if not isinstance(start_provenance, Mapping):
        raise ValueError("serialized correction-start provenance must be a mapping")
    is_vbd_provenance = start_provenance.get("contract") == "pss-solver-vbd-state-evidence-v1"
    is_vbd_role = start_role.startswith("vbd-k")
    if is_vbd_role != is_vbd_provenance:
        raise ValueError("serialized reserved vbd-k role and VBD evidence contract disagree")
    if is_vbd_provenance:
        if start_vbd_result is None:
            raise ValueError("serialized VBD correction start requires its original run result")
        validated_start = build_vbd_correction_start(
            scene,
            problem,
            start_vbd_result,
            reference,
        )
        if _canonical_digest(validated_start.as_dict()) != _canonical_digest(dict(start)):
            raise ValueError("serialized VBD correction start changed from its original run result")
    elif start_vbd_result is not None:
        raise ValueError("serialized non-VBD correction start cannot carry a VBD run result")
    else:
        validated_start = build_correction_start(
            scene,
            problem,
            start_array,
            reference,
            role=start_role,
            provenance=start_provenance,
        )
        if _canonical_digest(validated_start.as_dict()) != _canonical_digest(dict(start)):
            raise ValueError("serialized generic correction start changed from canonical reconstruction")

    solver = _freeze_json(record.get("solver"))
    if not isinstance(solver, Mapping):
        raise ValueError("serialized residual-Newton record is missing")
    _validate_solver_record(solver, config, requested)
    trace = solver["trace"]
    first = trace[0]
    final = trace[-1]
    for recorded, measured, label in (
        (first["objective"], initial_metrics.objective, "initial objective"),
        (first["gradient_norm"], initial_metrics.gradient_norm, "initial gradient"),
        (first["relative_residual"], initial_metrics.relative_residual, "initial residual"),
        (final["objective"], final_metrics.objective, "final objective"),
        (final["gradient_norm"], final_metrics.gradient_norm, "final gradient"),
        (final["relative_residual"], final_metrics.relative_residual, "final residual"),
    ):
        if recorded != measured:
            raise ValueError(f"serialized independent metrics disagree with the {label}")
    canonical_problem, _, _ = _validated_canonical_problem(scene, problem)
    replay = solve_newton_residual_polish(canonical_problem, start_array, config)
    if _array_digest(replay.x.detach().numpy()) != _array_digest(endpoint_array):
        raise ValueError("serialized correction endpoint does not match deterministic solver replay")
    if _canonical_digest(replay.deterministic_record()) != _canonical_digest(_thaw_json(solver)):
        raise ValueError("serialized residual-Newton record does not match deterministic solver replay")
    applied = int(solver["accepted_iterations"])
    exact = solver["reason"] in ("max_iterations", "gradient") and applied == requested
    saturated = solver["reason"] == "gradient" and bool(solver["converged"]) and applied < requested
    state_valid = (
        final_metrics.max_pin_error_m == 0.0
        and final_metrics.inverted_tet_fraction == 0.0
        and final_metrics.determinant_min > 0.0
        and final_metrics.minimum_singular_value > 0.0
    )
    if type(record.get("applied_corrections")) is not int or record["applied_corrections"] != applied:
        raise ValueError("serialized applied-correction count is inconsistent")
    if type(record.get("exact_budget_completed")) is not bool or record["exact_budget_completed"] != exact:
        raise ValueError("serialized exact-budget decision is inconsistent")
    if type(record.get("saturated")) is not bool or record["saturated"] != saturated:
        raise ValueError("serialized saturation decision is inconsistent")
    if type(record.get("state_valid")) is not bool or record["state_valid"] != state_valid:
        raise ValueError("serialized endpoint-validity decision is inconsistent")
