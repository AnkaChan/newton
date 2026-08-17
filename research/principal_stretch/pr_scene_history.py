# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Auditable dense-Newton histories for the PR #2901 driven tet scenes.

The original examples apply a Python callback once per rendered frame and
then execute five VBD substeps.  A history transition therefore distinguishes
three states:

* ``C_k`` is the committed float32 position, velocity, and flag state.
* ``A_k`` is ``C_k`` after the frame callback has overwritten driven pins or
  changed their active flags.
* ``T_k`` solves one common-objective implicit step from ``C_k``, with
  ``A_k`` supplying the actual Dirichlet targets and flags.

Accepted dense float64 Newton positions are committed to float32.  Velocities
are then evaluated as ``(q_next - A_k.q) / float32(dt)`` in float32, matching
the operation order of VBD's ``update_velocity`` kernel.  Content hashes never
contain timings; timing records are kept in a parallel tuple.

This is research infrastructure, not a public Newton API.  The default range
contains one atomic substep and the default safety cap is eight substeps, so a
caller cannot accidentally launch a 2,000-solve history.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import struct
import types
from collections.abc import Mapping

import numpy as np

import newton

from .newton_baseline import NewtonConfig, NewtonResidualPolishConfig
from .solver_benchmark import (
    NewtonAlternateResidualVerification,
    NewtonResidualReferenceRecovery,
    NewtonRunResult,
    TetBenchmarkScene,
    _newton_result_deterministic_record,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
    recover_newton_reference_with_residual_polish,
    run_newton,
    verify_newton_alternate_start_with_residual_polish,
)
from .solver_scenes import build_compression_scene, build_stretch_scene, build_twist_scene

_SCHEMA_VERSION = 2
_SOURCE_REVISION = "a513d446e42477a8ada78070f92ffb60d3108eeb"
_SUBSTEPS_PER_FRAME = 5
_TOTAL_FRAMES = 400
_DT = float(np.float32(1.0 / 300.0))
_ACTIVE_FLAG = int(newton.ParticleFlags.ACTIVE)
_KINDS = ("stretch", "twist", "compression-50", "compression-90")
_STALLED_RETRY_POLICY = "stalled-step-relative-tolerance-zero-v1"
_RESIDUAL_RECOVERY_RETRY_POLICY = "strict-reference-zero-step-prerequisite-v1"
_RESIDUAL_POLISH_POLICY = "strict-reference-residual-newton-three-start-v1"
_ALTERNATE_RECOVERY_RETRY_POLICY = "alternate-residual-zero-step-prerequisite-v1"
_ALTERNATE_RESIDUAL_POLICY = "alternate-start-only-residual-verification-v1"
_TRAINING_STATIC_ARRAY_NAMES = (
    "rest_q",
    "tet_indices",
    "tet_poses",
    "mass",
    "tet_materials",
    "gravity",
    "external_force",
)


def _readonly_array(value, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _canonical_array(value: np.ndarray) -> np.ndarray:
    dtype = value.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    return np.ascontiguousarray(value, dtype=canonical_dtype)


def _array_digest(value: np.ndarray) -> str:
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return types.MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validated_json_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    try:
        copied = json.loads(json.dumps(_thaw_json(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    return _freeze_json(copied)


def _float32_bits(value: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', np.float32(value)))[0]:08x}"


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "dtype": value.dtype.name,
        "shape": list(value.shape),
        "sha256": _array_digest(value),
    }


@dataclasses.dataclass(frozen=True, order=True)
class AtomicCoordinate:
    """Frame and substep coordinate of one committed state ``C_k``."""

    frame: int
    substep: int

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, numbers.Integral) or self.frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if (
            isinstance(self.substep, bool)
            or not isinstance(self.substep, numbers.Integral)
            or not 0 <= self.substep < _SUBSTEPS_PER_FRAME
        ):
            raise ValueError(f"substep must lie in [0, {_SUBSTEPS_PER_FRAME})")
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(self, "substep", int(self.substep))

    @property
    def ordinal(self) -> int:
        """Zero-based atomic-substep index."""
        return self.frame * _SUBSTEPS_PER_FRAME + self.substep

    def next(self) -> AtomicCoordinate:
        """Return the coordinate following one atomic substep."""
        if self.substep + 1 < _SUBSTEPS_PER_FRAME:
            return AtomicCoordinate(self.frame, self.substep + 1)
        return AtomicCoordinate(self.frame + 1, 0)

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-compatible coordinate."""
        return {"frame": self.frame, "substep": self.substep, "ordinal": self.ordinal}

    @classmethod
    def from_ordinal(cls, ordinal: int) -> AtomicCoordinate:
        """Construct a coordinate from an atomic-substep index."""
        if isinstance(ordinal, bool) or not isinstance(ordinal, numbers.Integral) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        frame, substep = divmod(int(ordinal), _SUBSTEPS_PER_FRAME)
        return cls(frame, substep)


@dataclasses.dataclass(frozen=True)
class FrameSchedule:
    """Exact callback schedule for one PR frame."""

    frame: int
    action: str
    value_name: str | None
    value: float | None
    released: bool

    def as_dict(self) -> dict[str, object]:
        """Return the deterministic schedule record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, eq=False)
class PRHistoryManifest:
    """Immutable physical and schedule identity of one PR history."""

    kind: str
    source_path: str
    base_physical_sha256: str
    topology_sha256: str
    material_sha256: str
    compression_ratio: float | None
    schedule: Mapping[str, object]
    total_frames: int = _TOTAL_FRAMES
    substeps_per_frame: int = _SUBSTEPS_PER_FRAME
    dt_seconds: float = _DT
    schema_version: int = _SCHEMA_VERSION
    source_revision: str = _SOURCE_REVISION
    manifest_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}")
        if self.total_frames != _TOTAL_FRAMES or self.substeps_per_frame != _SUBSTEPS_PER_FRAME:
            raise ValueError("PR history uses exactly 400 frames and five substeps per frame")
        canonical_dt = float(np.float32(self.dt_seconds))
        if canonical_dt != _DT:
            raise ValueError("PR history uses the exact float32 1/300 second substep")
        object.__setattr__(self, "dt_seconds", canonical_dt)
        object.__setattr__(self, "schedule", _validated_json_mapping(self.schedule, "schedule"))
        object.__setattr__(self, "manifest_sha256", _canonical_digest(self._payload()))

    @property
    def transition_count(self) -> int:
        """Number of atomic substeps in the complete 400-frame history."""
        return self.total_frames * self.substeps_per_frame

    @property
    def end_coordinate(self) -> AtomicCoordinate:
        """Exclusive endpoint of the complete history."""
        return AtomicCoordinate(self.total_frames, 0)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": "pr2901-callback-dense-newton-history-v2",
            "kind": self.kind,
            "source_path": self.source_path,
            "source_revision": self.source_revision,
            "base_physical_sha256": self.base_physical_sha256,
            "topology_sha256": self.topology_sha256,
            "material_sha256": self.material_sha256,
            "compression_ratio": self.compression_ratio,
            "total_frames": self.total_frames,
            "substeps_per_frame": self.substeps_per_frame,
            "dt_seconds": self.dt_seconds,
            "dt_float32_bits": _float32_bits(self.dt_seconds),
            "schedule": _thaw_json(self.schedule),
            "callback_order": "C_k -> frame callback at substep 0 -> A_k -> T_k",
            "commit_order": "q_next=float32(x_newton64); qd_next=float32((q_next-A_k.q)/dt32)",
            "reference_gate": {
                "stages": [
                    "solver_benchmark.run_newton primary",
                    "conditional zero-step-tolerance Newton retry",
                    "conditional strict three-start residual polish",
                    "conditional alternate-start-only residual verification",
                ],
                "gradient_limit_N": "max(1e-10, 1e-10*residual_scale)",
                "verification_displacement_relative_max": 1.0e-12,
                "alternate_displacement_relative_max": 1.0e-9,
                "requires_exact_pins": True,
                "requires_finite_uninverted_state": True,
            },
        }

    def as_dict(self) -> dict[str, object]:
        """Return the self-checking JSON-compatible manifest."""
        payload = self._payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload


@dataclasses.dataclass(frozen=True, eq=False)
class PRHistoryStaticBundle:
    """Immutable mesh, material, and load arrays shared by every transition."""

    manifest_sha256: str
    base_physical_sha256: str
    topology_sha256: str
    material_sha256: str
    rest_q: np.ndarray
    tet_indices: np.ndarray
    tet_poses: np.ndarray
    mass: np.ndarray
    tet_materials: np.ndarray
    gravity: np.ndarray
    external_force: np.ndarray
    static_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        arrays = {
            "rest_q": _readonly_array(self.rest_q, np.float32, "static rest_q"),
            "tet_indices": _readonly_array(self.tet_indices, np.int64, "static tet_indices"),
            "tet_poses": _readonly_array(self.tet_poses, np.float32, "static tet_poses"),
            "mass": _readonly_array(self.mass, np.float32, "static mass"),
            "tet_materials": _readonly_array(self.tet_materials, np.float32, "static tet_materials"),
            "gravity": _readonly_array(self.gravity, np.float32, "static gravity"),
            "external_force": _readonly_array(self.external_force, np.float32, "static external_force"),
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

        n_vertices = self.rest_q.shape[0]
        n_tets = self.tet_indices.shape[0]
        expected_shapes = {
            "rest_q": (n_vertices, 3),
            "tet_indices": (n_tets, 4),
            "tet_poses": (n_tets, 3, 3),
            "mass": (n_vertices,),
            "tet_materials": (n_tets, 3),
            "gravity": (3,),
            "external_force": (n_vertices, 3),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"static {name} must have shape {shape}")
        if n_vertices == 0 or n_tets == 0:
            raise ValueError("static history bundle must contain vertices and tetrahedra")
        if self.tet_indices.min() < 0 or self.tet_indices.max() >= n_vertices:
            raise ValueError("static tet_indices contains an out-of-range vertex")
        object.__setattr__(self, "static_sha256", _canonical_digest(self._payload()))

    def training_arrays(self) -> dict[str, np.ndarray]:
        """Return the canonical static arrays in the common-trainer order."""
        return {name: getattr(self, name) for name in _TRAINING_STATIC_ARRAY_NAMES}

    def _payload(self) -> dict[str, object]:
        return {
            "contract": "pr2901-history-training-static-v1",
            "manifest_sha256": self.manifest_sha256,
            "base_physical_sha256": self.base_physical_sha256,
            "topology_sha256": self.topology_sha256,
            "material_sha256": self.material_sha256,
            "arrays": {name: _array_record(value) for name, value in self.training_arrays().items()},
        }

    def as_dict(self) -> dict[str, object]:
        """Return self-checking hashes for the shared training arrays."""
        payload = self._payload()
        payload["static_sha256"] = self.static_sha256
        return payload


@dataclasses.dataclass(frozen=True, eq=False)
class CommittedState:
    """Canonical float32 state committed at an atomic coordinate."""

    manifest_sha256: str
    coordinate: AtomicCoordinate
    q: np.ndarray
    qd: np.ndarray
    particle_flags: np.ndarray
    state_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        q = _readonly_array(self.q, np.float32, "q")
        qd = _readonly_array(self.qd, np.float32, "qd")
        flags = _readonly_array(self.particle_flags, np.int32, "particle_flags")
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("q must have shape (V, 3)")
        if qd.shape != q.shape or flags.shape != (q.shape[0],):
            raise ValueError("qd and particle_flags do not match q")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", qd)
        object.__setattr__(self, "particle_flags", flags)
        object.__setattr__(self, "state_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "coordinate": self.coordinate.as_dict(),
            "q": _array_record(self.q),
            "qd": _array_record(self.qd),
            "particle_flags": _array_record(self.particle_flags),
        }

    def as_dict(self) -> dict[str, object]:
        """Return the state content record without raw arrays."""
        payload = self._payload()
        payload["state_sha256"] = self.state_sha256
        return payload


@dataclasses.dataclass(frozen=True, eq=False)
class AppliedAtomicState:
    """State ``A_k`` after the once-per-frame callback."""

    manifest_sha256: str
    coordinate: AtomicCoordinate
    input_state_sha256: str
    callback_applied: bool
    action: str
    schedule_value_name: str | None
    schedule_value: float | None
    q: np.ndarray
    particle_flags: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray
    applied_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        q = _readonly_array(self.q, np.float32, "applied q")
        flags = _readonly_array(self.particle_flags, np.int32, "applied particle_flags")
        pinned = _readonly_array(self.pinned_indices, np.int64, "pinned_indices")
        targets = _readonly_array(self.pin_targets, np.float32, "pin_targets")
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("applied q must have shape (V, 3)")
        if flags.shape != (q.shape[0],):
            raise ValueError("applied flags do not match q")
        if pinned.ndim != 1 or targets.shape != (pinned.size, 3):
            raise ValueError("applied pin targets do not match pinned_indices")
        if pinned.size and not np.array_equal(pinned, np.unique(pinned)):
            raise ValueError("applied pinned_indices must be sorted and unique")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "particle_flags", flags)
        object.__setattr__(self, "pinned_indices", pinned)
        object.__setattr__(self, "pin_targets", targets)
        object.__setattr__(self, "applied_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "coordinate": self.coordinate.as_dict(),
            "input_state_sha256": self.input_state_sha256,
            "callback_applied": self.callback_applied,
            "action": self.action,
            "schedule_value_name": self.schedule_value_name,
            "schedule_value": self.schedule_value,
            "q": _array_record(self.q),
            "particle_flags": _array_record(self.particle_flags),
            "pinned_indices": _array_record(self.pinned_indices),
            "pin_targets": _array_record(self.pin_targets),
        }

    def as_dict(self) -> dict[str, object]:
        """Return the callback content record without raw arrays."""
        payload = self._payload()
        payload["applied_sha256"] = self.applied_sha256
        return payload


def _root_prefix(manifest_sha256: str, state_sha256: str) -> str:
    return _canonical_digest(
        {
            "contract": "pr2901-history-prefix-root-v1",
            "manifest_sha256": manifest_sha256,
            "state_sha256": state_sha256,
        }
    )


def _advance_prefix(prefix_sha256: str, transition_sha256: str) -> str:
    return _canonical_digest(
        {
            "contract": "pr2901-history-prefix-link-v1",
            "prefix_sha256": prefix_sha256,
            "transition_sha256": transition_sha256,
        }
    )


@dataclasses.dataclass(frozen=True, eq=False)
class HistoryCheckpoint:
    """Content-addressed resumable history state."""

    manifest_sha256: str
    state: CommittedState
    prior_transition_sha256: str | None
    prefix_sha256: str
    checkpoint_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.state.manifest_sha256 != self.manifest_sha256:
            raise ValueError("checkpoint state belongs to a different manifest")
        object.__setattr__(self, "checkpoint_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "state_sha256": self.state.state_sha256,
            "coordinate": self.state.coordinate.as_dict(),
            "prior_transition_sha256": self.prior_transition_sha256,
            "prefix_sha256": self.prefix_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        """Return the self-checking checkpoint record."""
        payload = self._payload()
        payload["checkpoint_sha256"] = self.checkpoint_sha256
        return payload

    def training_arrays(self) -> dict[str, np.ndarray]:
        """Expose the pre-event state arrays needed to resume data generation."""
        return {
            "x_current": self.state.q,
            "velocity": self.state.qd,
            "particle_flags": self.state.particle_flags,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class _ReferenceStep:
    positions: np.ndarray
    accepted: bool
    failures: tuple[str, ...]
    deterministic_record: Mapping[str, object]
    timing_record: Mapping[str, object]
    newton_run: NewtonRunResult | None = dataclasses.field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _readonly_array(self.positions, np.float64, "reference positions"))
        object.__setattr__(
            self,
            "deterministic_record",
            _validated_json_mapping(self.deterministic_record, "reference deterministic_record"),
        )
        object.__setattr__(self, "timing_record", _validated_json_mapping(self.timing_record, "timing_record"))
        object.__setattr__(self, "failures", tuple(str(item) for item in self.failures))


@dataclasses.dataclass(frozen=True, eq=False)
class HistoryTransition:
    """One accepted, content-addressed atomic transition ``T_k``."""

    manifest_sha256: str
    coordinate: AtomicCoordinate
    next_coordinate: AtomicCoordinate
    input_state_sha256: str
    input_prefix_sha256: str
    input_state: CommittedState
    applied_state: AppliedAtomicState
    applied_record: Mapping[str, object]
    scene_sha256: str
    objective_instance_sha256: str
    dt_seconds: float
    topology_sha256: str
    material_sha256: str
    inertial_target: np.ndarray
    reference_record: Mapping[str, object]
    reference_positions: np.ndarray
    output_state: CommittedState
    transition_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.next_coordinate != self.coordinate.next():
            raise ValueError("transition coordinates are not adjacent")
        if self.output_state.coordinate != self.next_coordinate:
            raise ValueError("transition output has the wrong coordinate")
        if self.output_state.manifest_sha256 != self.manifest_sha256:
            raise ValueError("transition output belongs to a different manifest")
        if self.input_state.manifest_sha256 != self.manifest_sha256:
            raise ValueError("transition input belongs to a different manifest")
        if self.input_state.coordinate != self.coordinate:
            raise ValueError("transition input has the wrong coordinate")
        if self.input_state.state_sha256 != self.input_state_sha256:
            raise ValueError("transition input state hash does not match its arrays")
        if self.applied_state.manifest_sha256 != self.manifest_sha256:
            raise ValueError("transition applied state belongs to a different manifest")
        if self.applied_state.coordinate != self.coordinate:
            raise ValueError("transition applied state has the wrong coordinate")
        if self.applied_state.input_state_sha256 != self.input_state_sha256:
            raise ValueError("transition applied state is disconnected from its input")
        if self.applied_state.as_dict() != _thaw_json(self.applied_record):
            raise ValueError("transition applied record does not match its arrays")
        canonical_dt = float(np.float32(self.dt_seconds))
        if canonical_dt != _DT:
            raise ValueError("transition dt does not match the exact PR substep")
        object.__setattr__(self, "dt_seconds", canonical_dt)
        inertial_target = _readonly_array(self.inertial_target, np.float32, "inertial_target")
        if inertial_target.shape != self.output_state.q.shape:
            raise ValueError("inertial_target does not match the output state")
        if not np.array_equal(
            inertial_target[self.applied_state.pinned_indices],
            self.applied_state.pin_targets,
        ):
            raise ValueError("transition inertial target does not preserve applied pins")
        object.__setattr__(self, "inertial_target", inertial_target)
        reference_positions = _readonly_array(self.reference_positions, np.float64, "reference_positions")
        if reference_positions.shape != self.output_state.q.shape:
            raise ValueError("reference_positions do not match the output state")
        object.__setattr__(self, "reference_positions", reference_positions)
        object.__setattr__(self, "applied_record", _validated_json_mapping(self.applied_record, "applied_record"))
        reference_record = _validated_json_mapping(self.reference_record, "reference_record")
        object.__setattr__(
            self,
            "reference_record",
            reference_record,
        )

        expected_q = np.asarray(reference_positions, dtype=np.float32)
        if not np.array_equal(self.output_state.q, expected_q):
            raise ValueError("transition output q is not the float32 committed reference")
        if not np.array_equal(self.output_state.particle_flags, self.applied_state.particle_flags):
            raise ValueError("transition output flags do not match the applied callback state")
        expected_qd = (expected_q - self.applied_state.q) / np.float32(canonical_dt)
        expected_qd = np.asarray(expected_qd, dtype=np.float32)
        expected_qd[self.applied_state.pinned_indices] = np.float32(0.0)
        if not np.array_equal(self.output_state.qd, expected_qd):
            raise ValueError("transition output velocity does not match the exact float32 commit formula")
        recorded_position_sha256 = reference_record.get("position_sha256")
        if recorded_position_sha256 is not None and recorded_position_sha256 != _array_digest(reference_positions):
            raise ValueError("reference record position hash does not match reference_positions")
        object.__setattr__(self, "transition_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "coordinate": self.coordinate.as_dict(),
            "next_coordinate": self.next_coordinate.as_dict(),
            "input_state_sha256": self.input_state_sha256,
            "input_prefix_sha256": self.input_prefix_sha256,
            "applied_record": _thaw_json(self.applied_record),
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "dt_seconds": self.dt_seconds,
            "topology_sha256": self.topology_sha256,
            "material_sha256": self.material_sha256,
            "inertial_target": _array_record(self.inertial_target),
            "reference_record": _thaw_json(self.reference_record),
            "reference_positions": _array_record(self.reference_positions),
            "output_state_sha256": self.output_state.state_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        """Return the transition content record without raw arrays."""
        payload = self._payload()
        payload["transition_sha256"] = self.transition_sha256
        return payload

    def model_inputs(self) -> dict[str, np.ndarray]:
        """Return the exact dynamic arrays consumed by the common v3 trainer.

        The model observes post-callback ``A_k.q`` as its current positions.
        Its previous positions follow the committed-velocity contract
        ``C_k.q - float32(dt) * C_k.qd``; multiplication, subtraction, and the
        final cast are all performed in float32 and in that order.  Gravity,
        loads, material coefficients, and rest topology live once in
        :class:`PRHistoryStaticBundle` rather than being duplicated here.
        """
        dt32 = np.float32(self.dt_seconds)
        displacement = (self.input_state.qd * dt32).astype(np.float32)
        x_previous = (self.input_state.q - displacement).astype(np.float32)
        x_previous = _readonly_array(x_previous, np.float32, "model x_previous")
        return {
            "x_current": self.applied_state.q,
            "x_previous": x_previous,
            "pinned_indices": self.applied_state.pinned_indices,
            "pin_targets": self.applied_state.pin_targets,
            "inertial_target": self.inertial_target,
        }

    def training_arrays(self) -> dict[str, np.ndarray]:
        """Expose a self-contained sample for a separate learned solver.

        The input arrays are pre-event ``C_k`` values.  ``x_applied`` and the
        dynamic pin arrays describe ``A_k``; this includes compression release
        transitions where the top face is absent from ``pinned_indices``.
        """
        return {
            "x_current": self.input_state.q,
            "velocity": self.input_state.qd,
            "particle_flags_pre_event": self.input_state.particle_flags,
            "x_applied": self.applied_state.q,
            "particle_flags_applied": self.applied_state.particle_flags,
            "pinned_indices": self.applied_state.pinned_indices,
            "pin_targets": self.applied_state.pin_targets,
            "inertial_target": self.inertial_target,
            "x_reference": self.reference_positions,
            "x_committed": self.output_state.q,
            "velocity_committed": self.output_state.qd,
        }

    def training_record(self) -> dict[str, object]:
        """Return hashes and scalars paired with :meth:`training_arrays`."""
        return {
            "manifest_sha256": self.manifest_sha256,
            "transition_sha256": self.transition_sha256,
            "coordinate": self.coordinate.as_dict(),
            "dt_seconds": self.dt_seconds,
            "topology_sha256": self.topology_sha256,
            "material_sha256": self.material_sha256,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "model_inputs": {name: _array_record(value) for name, value in self.model_inputs().items()},
            "arrays": {name: _array_record(value) for name, value in self.training_arrays().items()},
        }


@dataclasses.dataclass(frozen=True, eq=False)
class FailedReference:
    """Fail-closed record for a rejected dense-Newton reference."""

    manifest_sha256: str
    coordinate: AtomicCoordinate
    input_state_sha256: str
    input_prefix_sha256: str
    applied_record: Mapping[str, object]
    scene_sha256: str
    objective_instance_sha256: str
    reference_record: Mapping[str, object]
    failures: tuple[str, ...]
    failure_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if not self.failures:
            raise ValueError("a failed reference must report at least one failure")
        object.__setattr__(self, "failures", tuple(str(item) for item in self.failures))
        object.__setattr__(self, "applied_record", _validated_json_mapping(self.applied_record, "applied_record"))
        object.__setattr__(
            self,
            "reference_record",
            _validated_json_mapping(self.reference_record, "reference_record"),
        )
        object.__setattr__(self, "failure_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "coordinate": self.coordinate.as_dict(),
            "input_state_sha256": self.input_state_sha256,
            "input_prefix_sha256": self.input_prefix_sha256,
            "applied_record": _thaw_json(self.applied_record),
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "reference_record": _thaw_json(self.reference_record),
            "failures": list(self.failures),
        }

    def as_dict(self) -> dict[str, object]:
        """Return the self-checking failure record."""
        payload = self._payload()
        payload["failure_sha256"] = self.failure_sha256
        return payload


@dataclasses.dataclass(frozen=True, eq=False)
class TransitionTiming:
    """Timing-only record, deliberately excluded from chain content hashes."""

    coordinate: AtomicCoordinate
    accepted: bool
    values: Mapping[str, object]
    timing_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _validated_json_mapping(self.values, "timing values"))
        payload = {
            "coordinate": self.coordinate.as_dict(),
            "accepted": self.accepted,
            "values": _thaw_json(self.values),
        }
        object.__setattr__(self, "timing_sha256", _canonical_digest(payload))


@dataclasses.dataclass(frozen=True, eq=False)
class PRHistoryChain:
    """A verified accepted range, optionally terminated by reference failure."""

    manifest: PRHistoryManifest
    initial_checkpoint: HistoryCheckpoint
    transitions: tuple[HistoryTransition, ...]
    timings: tuple[TransitionTiming, ...]
    final_checkpoint: HistoryCheckpoint
    termination: str
    failed_reference: FailedReference | None = None
    prior_chain_sha256: str | None = None
    chain_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "timings", tuple(self.timings))
        self.verify()
        payload = {
            "manifest_sha256": self.manifest.manifest_sha256,
            "initial_checkpoint_sha256": self.initial_checkpoint.checkpoint_sha256,
            "transition_sha256": [item.transition_sha256 for item in self.transitions],
            "final_checkpoint_sha256": self.final_checkpoint.checkpoint_sha256,
            "termination": self.termination,
            "failed_reference_sha256": (
                None if self.failed_reference is None else self.failed_reference.failure_sha256
            ),
            "prior_chain_sha256": self.prior_chain_sha256,
        }
        object.__setattr__(self, "chain_sha256", _canonical_digest(payload))

    def verify(self) -> None:
        """Reject cross-manifest, reordered, disconnected, or tampered chains."""
        manifest_sha256 = self.manifest.manifest_sha256
        if self.initial_checkpoint.manifest_sha256 != manifest_sha256:
            raise ValueError("initial checkpoint belongs to a different manifest")
        if self.final_checkpoint.manifest_sha256 != manifest_sha256:
            raise ValueError("final checkpoint belongs to a different manifest")

        current_state = self.initial_checkpoint.state
        current_prefix = self.initial_checkpoint.prefix_sha256
        prior_transition = self.initial_checkpoint.prior_transition_sha256
        if current_state.coordinate == AtomicCoordinate(0, 0):
            if self.prior_chain_sha256 is not None:
                raise ValueError("a root history chain cannot declare a prior-chain proof")
            if prior_transition is not None:
                raise ValueError("initial history state cannot have a prior transition")
            expected_root = _root_prefix(manifest_sha256, current_state.state_sha256)
            if current_prefix != expected_root:
                raise ValueError("initial history prefix is invalid")
        else:
            if (
                not isinstance(self.prior_chain_sha256, str)
                or len(self.prior_chain_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.prior_chain_sha256)
            ):
                raise ValueError("a resumed history chain requires a verified prior-chain SHA-256 proof")

        for transition in self.transitions:
            if transition.manifest_sha256 != manifest_sha256:
                raise ValueError("transition belongs to a different manifest")
            if transition.dt_seconds != self.manifest.dt_seconds:
                raise ValueError("transition dt does not match the manifest")
            if transition.topology_sha256 != self.manifest.topology_sha256:
                raise ValueError("transition topology does not match the manifest")
            if transition.material_sha256 != self.manifest.material_sha256:
                raise ValueError("transition material does not match the manifest")
            if transition.coordinate != current_state.coordinate:
                raise ValueError("transition order is disconnected")
            if transition.input_state_sha256 != current_state.state_sha256:
                raise ValueError("transition input state is disconnected")
            if transition.input_prefix_sha256 != current_prefix:
                raise ValueError("transition prefix is disconnected")
            current_prefix = _advance_prefix(current_prefix, transition.transition_sha256)
            current_state = transition.output_state
            prior_transition = transition.transition_sha256

        if self.final_checkpoint.state.state_sha256 != current_state.state_sha256:
            raise ValueError("final checkpoint state does not match the accepted chain")
        if self.final_checkpoint.prefix_sha256 != current_prefix:
            raise ValueError("final checkpoint prefix does not match the accepted chain")
        if self.final_checkpoint.prior_transition_sha256 != prior_transition:
            raise ValueError("final checkpoint predecessor does not match the accepted chain")

        expected_timing_count = len(self.transitions) + int(self.failed_reference is not None)
        if len(self.timings) != expected_timing_count:
            raise ValueError("timing records do not match transition attempts")
        for timing, transition in zip(self.timings, self.transitions, strict=False):
            if timing.coordinate != transition.coordinate or not timing.accepted:
                raise ValueError("accepted transition timing records are out of order")

        if self.failed_reference is None:
            if self.termination not in ("range_complete", "history_complete"):
                raise ValueError("a successful chain has an invalid termination")
        else:
            failure = self.failed_reference
            if self.termination != "failed_reference":
                raise ValueError("a rejected reference must terminate the chain")
            if failure.manifest_sha256 != manifest_sha256:
                raise ValueError("failed reference belongs to a different manifest")
            if failure.coordinate != current_state.coordinate:
                raise ValueError("failed reference is disconnected from the chain")
            if failure.input_state_sha256 != current_state.state_sha256:
                raise ValueError("failed reference input state is disconnected")
            if failure.input_prefix_sha256 != current_prefix:
                raise ValueError("failed reference prefix is disconnected")
            timing = self.timings[-1]
            if timing.coordinate != failure.coordinate or timing.accepted:
                raise ValueError("failed reference timing record is out of order")

    def checkpoint_at(self, coordinate: AtomicCoordinate) -> HistoryCheckpoint:
        """Select a verified checkpoint from the accepted range."""
        if coordinate == self.initial_checkpoint.state.coordinate:
            return self.initial_checkpoint
        prefix = self.initial_checkpoint.prefix_sha256
        for transition in self.transitions:
            prefix = _advance_prefix(prefix, transition.transition_sha256)
            if transition.output_state.coordinate == coordinate:
                return HistoryCheckpoint(
                    manifest_sha256=self.manifest.manifest_sha256,
                    state=transition.output_state,
                    prior_transition_sha256=transition.transition_sha256,
                    prefix_sha256=prefix,
                )
        raise ValueError("coordinate is not an accepted checkpoint in this chain")

    def as_dict(self) -> dict[str, object]:
        """Return the complete records, with timings in a separate section."""
        return {
            "manifest": self.manifest.as_dict(),
            "initial_checkpoint": self.initial_checkpoint.as_dict(),
            "transitions": [item.as_dict() for item in self.transitions],
            "final_checkpoint": self.final_checkpoint.as_dict(),
            "termination": self.termination,
            "failed_reference": None if self.failed_reference is None else self.failed_reference.as_dict(),
            "prior_chain_sha256": self.prior_chain_sha256,
            "verification_scope": {
                "starts_at_root": self.initial_checkpoint.state.coordinate == AtomicCoordinate(0, 0),
                "prior_history_proof": self.prior_chain_sha256,
            },
            "chain_sha256": self.chain_sha256,
            "timings": [
                {
                    "coordinate": item.coordinate.as_dict(),
                    "accepted": item.accepted,
                    "values": _thaw_json(item.values),
                    "timing_sha256": item.timing_sha256,
                }
                for item in self.timings
            ],
        }


def _base_physical_digest(scene: TetBenchmarkScene) -> str:
    arrays = {}
    for name in (
        "rest_q",
        "tet_indices",
        "tet_poses",
        "mass",
        "particle_inv_mass",
        "tet_materials",
        "tri_indices",
        "tri_poses",
        "tri_materials",
        "tri_areas",
        "particle_flags",
        "color_group_offsets",
        "color_group_particles",
        "gravity",
        "external_force",
    ):
        arrays[name] = _array_record(getattr(scene, name))
    return _canonical_digest(
        {
            "contract": "pr2901-history-base-physical-v1",
            "source": scene.source,
            "dt_seconds": scene.dt,
            "coefficient_convention": scene.metadata["coefficient_convention"],
            "arrays": arrays,
        }
    )


def _topology_digest(scene: TetBenchmarkScene) -> str:
    return _canonical_digest(
        {
            "tet_indices": _array_record(scene.tet_indices),
            "tri_indices": _array_record(scene.tri_indices),
            "color_group_offsets": _array_record(scene.color_group_offsets),
            "color_group_particles": _array_record(scene.color_group_particles),
        }
    )


def _material_digest(scene: TetBenchmarkScene) -> str:
    return _canonical_digest(
        {
            "mass": _array_record(scene.mass),
            "particle_inv_mass": _array_record(scene.particle_inv_mass),
            "tet_materials": _array_record(scene.tet_materials),
            "tri_materials": _array_record(scene.tri_materials),
        }
    )


def _default_newton_config() -> NewtonConfig:
    return NewtonConfig(
        max_iterations=50,
        gradient_absolute_tolerance=1.0e-10,
        gradient_relative_tolerance=1.0e-10,
        step_relative_tolerance=1.0e-14,
    )


def _dense_reference_deterministic_record(run: NewtonRunResult) -> dict[str, object]:
    """Reconstruct every deterministic field serialized for one Newton run."""
    result = run.result
    return {
        "method": "dense-cpu-newton-float64",
        "config": dataclasses.asdict(run.config),
        "scene_sha256": run.scene_sha256,
        "objective_instance_sha256": run.objective_instance_sha256,
        "accepted": run.reference_accepted,
        "failures": list(run.reference_failures),
        "native_converged": result.converged,
        "native_reason": result.reason,
        "accepted_iterations": result.accepted_iterations,
        "final_objective": result.final_objective,
        "final_gradient_norm": result.final_gradient_norm,
        "final_relative_residual": result.final_relative_residual,
        "verification_displacement_relative": run.verification_displacement_relative,
        "verification_converged": run.verification_converged,
        "verification_reason": run.verification_reason,
        "alternate_start_displacement_relative": run.alternate_start_displacement_relative,
        "alternate_start_converged": run.alternate_start_converged,
        "alternate_start_reason": run.alternate_start_reason,
        "alternate_start_gradient_norm": run.alternate_start_gradient_norm,
        "alternate_start_relative_residual": run.alternate_start_relative_residual,
        "work": {
            "objective_evaluations": result.objective_evaluations,
            "gradient_evaluations": result.gradient_evaluations,
            "hessian_evaluations": result.hessian_evaluations,
            "eigenvalue_evaluations": result.eigenvalue_evaluations,
            "factorization_attempts": result.factorization_attempts,
            "line_search_trials": result.line_search_trials,
        },
        "trace": [
            {
                "iteration": item.iteration,
                "objective": item.objective,
                "gradient_norm": item.gradient_norm,
                "relative_residual": item.relative_residual,
                "accepted_step_norm": item.accepted_step_norm,
                "accepted_step_size": item.accepted_step_size,
                "regularization": item.regularization,
            }
            for item in result.trace
        ],
        "position_sha256": _array_digest(result.x.detach().numpy()),
    }


def _solve_dense_reference(
    scene: TetBenchmarkScene,
    config: NewtonConfig,
) -> _ReferenceStep:
    problem = build_common_problem(scene)
    objective = common_objective_manifest(scene, problem)
    run = run_newton(scene, problem, config=config, warmup=False, repeats=1)
    result = run.result
    deterministic_record = _dense_reference_deterministic_record(run)
    if objective["objective_instance_sha256"] != run.objective_instance_sha256:
        raise RuntimeError("dense reference changed the objective instance")
    timing_record = {
        "problem_setup_seconds": result.problem_setup_seconds,
        "residual_scale_setup_seconds": result.residual_scale_setup_seconds,
        "warmup_seconds": run.warmup_seconds,
        "repeat_seconds": list(run.repeat_seconds),
        "representative": {
            "total_seconds": result.total_seconds,
            "objective_gradient_seconds": result.objective_gradient_seconds,
            "hessian_seconds": result.hessian_seconds,
            "linear_solve_seconds": result.linear_solve_seconds,
            "line_search_seconds": result.line_search_seconds,
        },
        "trace_elapsed_seconds": [item.elapsed_seconds for item in result.trace],
    }
    return _ReferenceStep(
        positions=result.x.detach().numpy(),
        accepted=run.reference_accepted,
        failures=run.reference_failures,
        deterministic_record=deterministic_record,
        timing_record=timing_record,
        newton_run=run,
    )


def _stalled_reference_is_retryable(reference: _ReferenceStep, config: NewtonConfig) -> bool:
    """Return whether a rejected native stall is safe for the zero-step-tolerance retry."""
    if reference.accepted or config.step_relative_tolerance == 0.0:
        return False
    record = reference.deterministic_record
    if record.get("native_converged") is not False or record.get("native_reason") != "stalled":
        return False
    if _thaw_json(record.get("config")) != dataclasses.asdict(config):
        return False

    required_finite_scalars = (
        "final_objective",
        "final_gradient_norm",
        "final_relative_residual",
        "verification_displacement_relative",
        "alternate_start_displacement_relative",
        "alternate_start_gradient_norm",
        "alternate_start_relative_residual",
    )
    for name in required_finite_scalars:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            return False

    allowed_failures = (
        "native termination: stalled",
        "independent gradient ",
        "verification termination: stalled",
        "verification displacement ",
    )
    return allowed_failures[0] in reference.failures and all(
        failure == allowed_failures[0] or failure.startswith(allowed_failures[1:]) for failure in reference.failures
    )


def _retry_provenance_failures(
    primary: _ReferenceStep,
    retry: _ReferenceStep,
    retry_config: NewtonConfig,
) -> tuple[str, ...]:
    """Return deterministic provenance failures that prohibit selecting a retry."""
    failures = []
    record = retry.deterministic_record
    if _thaw_json(record.get("config")) != dataclasses.asdict(retry_config):
        failures.append("dense reference retry config does not match the requested retry config")

    for name in ("scene_sha256", "objective_instance_sha256"):
        primary_identity = primary.deterministic_record.get(name)
        retry_identity = record.get(name)
        if retry.accepted and (primary_identity is None or retry_identity is None):
            failures.append(f"accepted dense reference retry omitted {name}")
        elif retry_identity is not None and primary_identity != retry_identity:
            failures.append(f"dense reference retry changed {name}")

    if retry.accepted:
        if record.get("accepted") is not True:
            failures.append("accepted dense reference retry record does not declare acceptance")
        if record.get("position_sha256") != _array_digest(retry.positions):
            failures.append("accepted dense reference retry position hash does not match its positions")
    return tuple(failures)


def _combine_reference_attempts(
    primary: _ReferenceStep,
    retry: _ReferenceStep,
    provenance_failures: tuple[str, ...] = (),
    *,
    retry_policy: str = _STALLED_RETRY_POLICY,
) -> _ReferenceStep:
    """Bind both conditional-retry attempts and the selected attempt into their records."""
    selected_failures = provenance_failures or retry.failures
    selected_accepted = retry.accepted and not provenance_failures
    deterministic_record = dict(_thaw_json(retry.deterministic_record))
    deterministic_record.update(
        {
            "accepted": selected_accepted,
            "failures": list(selected_failures),
            "retry_policy": retry_policy,
            "selected_attempt": 1,
            "attempts": [
                {
                    "index": 0,
                    "role": "primary",
                    "record": _thaw_json(primary.deterministic_record),
                },
                {
                    "index": 1,
                    "role": "step-relative-tolerance-zero",
                    "record": _thaw_json(retry.deterministic_record),
                },
            ],
        }
    )
    if provenance_failures:
        deterministic_record["provenance_failures"] = list(provenance_failures)
    timing_record = dict(_thaw_json(retry.timing_record))
    timing_record.update(
        {
            "retry_policy": retry_policy,
            "selected_attempt": 1,
            "attempts": [
                {
                    "index": 0,
                    "role": "primary",
                    "record": _thaw_json(primary.timing_record),
                },
                {
                    "index": 1,
                    "role": "step-relative-tolerance-zero",
                    "record": _thaw_json(retry.timing_record),
                },
            ],
        }
    )
    return _ReferenceStep(
        positions=retry.positions,
        accepted=selected_accepted,
        failures=selected_failures,
        deterministic_record=deterministic_record,
        timing_record=timing_record,
        newton_run=retry.newton_run,
    )


def _solve_dense_reference_with_retry(
    scene: TetBenchmarkScene,
    config: NewtonConfig,
) -> _ReferenceStep:
    """Run the exact primary solve and conditionally retry one healthy native stall."""
    primary = _solve_dense_reference(scene, config)
    if not _stalled_reference_is_retryable(primary, config):
        return primary

    retry_config = dataclasses.replace(config, step_relative_tolerance=0.0)
    retry_config.validate()
    try:
        retry = _solve_dense_reference(scene, retry_config)
    except Exception as exc:
        message = f"dense reference stalled retry raised {type(exc).__name__}: {exc}"
        retry = _ReferenceStep(
            positions=primary.positions,
            accepted=False,
            failures=(message,),
            deterministic_record={
                "method": "dense-cpu-newton-float64",
                "config": dataclasses.asdict(retry_config),
                "accepted": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            timing_record={"unavailable_due_to_exception": True},
        )

    provenance_failures = _retry_provenance_failures(primary, retry, retry_config)
    return _combine_reference_attempts(primary, retry, provenance_failures)


def _reference_is_recovery_prerequisite(reference: _ReferenceStep, config: NewtonConfig) -> bool:
    """Allow one explicitly versioned zero-step prerequisite for residual polish."""
    if reference.accepted or config.step_relative_tolerance == 0.0:
        return False
    record = reference.deterministic_record
    native_reason = record.get("native_reason")
    if record.get("native_converged") is not False or native_reason not in ("stalled", "line_search"):
        return False
    if _thaw_json(record.get("config")) != dataclasses.asdict(config):
        return False
    required_finite_scalars = (
        "final_objective",
        "final_gradient_norm",
        "final_relative_residual",
        "verification_displacement_relative",
        "alternate_start_displacement_relative",
        "alternate_start_gradient_norm",
        "alternate_start_relative_residual",
    )
    for name in required_finite_scalars:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            return False
    allowed_failures = (
        "native termination: stalled",
        "native termination: line_search",
        "independent gradient ",
        "verification termination: stalled",
        "verification termination: line_search",
        "verification displacement ",
        "alternate-start gradient ",
    )
    native_failure = f"native termination: {native_reason}"
    return native_failure in reference.failures and all(
        failure in allowed_failures[:2] or failure.startswith(allowed_failures[2:]) for failure in reference.failures
    )


def _run_residual_recovery_zero_step_prerequisite(
    scene: TetBenchmarkScene,
    reference: _ReferenceStep,
    config: NewtonConfig,
) -> _ReferenceStep:
    """Run the staged zero-step prerequisite without changing the v1 retry."""
    if not _reference_is_recovery_prerequisite(reference, config):
        return reference
    retry_config = dataclasses.replace(config, step_relative_tolerance=0.0)
    retry_config.validate()
    try:
        retry = _solve_dense_reference(scene, retry_config)
    except Exception as exc:
        message = f"residual-recovery zero-step prerequisite raised {type(exc).__name__}: {exc}"
        retry = _ReferenceStep(
            positions=reference.positions,
            accepted=False,
            failures=(message,),
            deterministic_record={
                "method": "dense-cpu-newton-float64",
                "config": dataclasses.asdict(retry_config),
                "accepted": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            timing_record={"unavailable_due_to_exception": True},
        )
    provenance_failures = _retry_provenance_failures(reference, retry, retry_config)
    return _combine_reference_attempts(
        reference,
        retry,
        provenance_failures,
        retry_policy=_RESIDUAL_RECOVERY_RETRY_POLICY,
    )


def _reference_is_alternate_verification_prerequisite(
    reference: _ReferenceStep,
    config: NewtonConfig,
    problem,
) -> bool:
    """Recognize one valid representative with only an alternate-gradient failure."""
    if reference.accepted or config.step_relative_tolerance == 0.0 or reference.newton_run is None:
        return False
    record = reference.deterministic_record
    if _thaw_json(record.get("config")) != dataclasses.asdict(config):
        return False
    if record.get("accepted") is not False:
        return False
    if record.get("native_converged") is not True or record.get("native_reason") != "gradient":
        return False
    if record.get("verification_converged") is not True or record.get("verification_reason") != "gradient":
        return False
    if record.get("alternate_start_converged") is not False or record.get("alternate_start_reason") not in (
        "stalled",
        "line_search",
        "max_iterations",
    ):
        return False
    if len(reference.failures) != 1 or not reference.failures[0].startswith("alternate-start gradient "):
        return False
    required_finite_scalars = (
        "final_objective",
        "final_gradient_norm",
        "final_relative_residual",
        "verification_displacement_relative",
        "alternate_start_displacement_relative",
        "alternate_start_gradient_norm",
        "alternate_start_relative_residual",
    )
    if any(
        isinstance(record.get(name), bool)
        or not isinstance(record.get(name), numbers.Real)
        or not math.isfinite(float(record[name]))
        for name in required_finite_scalars
    ):
        return False
    residual_limit = max(1.0e-10, 1.0e-10 * problem.residual_scale)
    if (
        float(record["final_gradient_norm"]) > residual_limit
        or float(record["verification_displacement_relative"]) > 1.0e-12
        or float(record["alternate_start_displacement_relative"]) > 1.0e-9
        or float(record["alternate_start_gradient_norm"]) <= residual_limit
    ):
        return False
    if record.get("position_sha256") != _array_digest(reference.positions):
        return False
    try:
        metrics = evaluate_common_state(problem, reference.positions)
    except (RuntimeError, ValueError):
        return False
    return (
        metrics.gradient_norm <= residual_limit
        and metrics.max_pin_error_m == 0.0
        and metrics.inverted_tet_fraction == 0.0
    )


def _run_alternate_verification_zero_step_prerequisite(
    scene: TetBenchmarkScene,
    problem,
    reference: _ReferenceStep,
    config: NewtonConfig,
) -> _ReferenceStep:
    """Run a separately versioned zero-step prerequisite for alternate verification."""
    if not _reference_is_alternate_verification_prerequisite(reference, config, problem):
        return reference
    retry_config = dataclasses.replace(config, step_relative_tolerance=0.0)
    retry_config.validate()
    try:
        retry = _solve_dense_reference(scene, retry_config)
    except Exception as exc:
        message = f"alternate-verification zero-step prerequisite raised {type(exc).__name__}: {exc}"
        retry = _ReferenceStep(
            positions=reference.positions,
            accepted=False,
            failures=(message,),
            deterministic_record={
                "method": "dense-cpu-newton-float64",
                "config": dataclasses.asdict(retry_config),
                "accepted": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            timing_record={"unavailable_due_to_exception": True},
        )
    provenance_failures = _retry_provenance_failures(reference, retry, retry_config)
    return _combine_reference_attempts(
        reference,
        retry,
        provenance_failures,
        retry_policy=_ALTERNATE_RECOVERY_RETRY_POLICY,
    )


def _residual_polish_retry_config(
    reference: _ReferenceStep,
    primary_config: NewtonConfig,
    scene_sha256: str,
    objective_instance_sha256: str,
) -> NewtonConfig | None:
    """Return the authenticated zero-step retry config eligible for polish."""
    if reference.accepted:
        return None
    retry_config = dataclasses.replace(primary_config, step_relative_tolerance=0.0)
    retry_config.validate()
    record = _thaw_json(reference.deterministic_record)
    attempts = record.get("attempts")
    if (
        record.get("retry_policy") not in (_STALLED_RETRY_POLICY, _RESIDUAL_RECOVERY_RETRY_POLICY)
        or record.get("selected_attempt") != 1
        or record.get("accepted") is not False
        or record.get("provenance_failures")
        or not isinstance(attempts, list)
        or len(attempts) != 2
    ):
        return None
    primary_attempt, retry_attempt = attempts
    if (
        primary_attempt.get("index") != 0
        or primary_attempt.get("role") != "primary"
        or retry_attempt.get("index") != 1
        or retry_attempt.get("role") != "step-relative-tolerance-zero"
    ):
        return None
    primary_record = primary_attempt.get("record")
    retry_record = retry_attempt.get("record")
    if not isinstance(primary_record, dict) or not isinstance(retry_record, dict):
        return None
    if primary_record.get("config") != dataclasses.asdict(primary_config):
        return None
    if retry_record.get("config") != dataclasses.asdict(retry_config):
        return None
    if record.get("config") != dataclasses.asdict(retry_config):
        return None

    for identity, expected in (
        ("scene_sha256", scene_sha256),
        ("objective_instance_sha256", objective_instance_sha256),
    ):
        if expected is None or any(
            attempt.get(identity) != expected for attempt in (record, primary_record, retry_record)
        ):
            return None
    position_sha256 = _array_digest(reference.positions)
    if record.get("position_sha256") != position_sha256 or retry_record.get("position_sha256") != position_sha256:
        return None
    if retry_record.get("accepted") is not False:
        return None
    native_reason = retry_record.get("native_reason")
    if retry_record.get("native_converged") is not False or native_reason not in ("stalled", "line_search"):
        return None
    run = reference.newton_run
    if (
        run is None
        or run.config != retry_config
        or run.reference_accepted
        or run.result.converged
        or run.result.reason not in ("stalled", "line_search")
        or run.result_state_sha256 != position_sha256
        or _array_digest(run.result.x.detach().numpy()) != position_sha256
    ):
        return None
    if retry_record != _dense_reference_deterministic_record(run):
        return None

    required_finite_scalars = (
        "final_objective",
        "final_gradient_norm",
        "final_relative_residual",
        "verification_displacement_relative",
        "alternate_start_displacement_relative",
        "alternate_start_gradient_norm",
        "alternate_start_relative_residual",
    )
    if any(
        isinstance(retry_record.get(name), bool)
        or not isinstance(retry_record.get(name), numbers.Real)
        or not math.isfinite(float(retry_record[name]))
        for name in required_finite_scalars
    ):
        return None
    allowed_failures = (
        "native termination: stalled",
        "native termination: line_search",
        "independent gradient ",
        "verification termination: stalled",
        "verification termination: line_search",
        "verification displacement ",
        "alternate-start gradient ",
    )
    native_failure = f"native termination: {native_reason}"
    if native_failure not in reference.failures or not all(
        failure in allowed_failures[:2] or failure.startswith(allowed_failures[2:]) for failure in reference.failures
    ):
        return None
    return retry_config


def _alternate_verification_retry_config(
    reference: _ReferenceStep,
    primary_config: NewtonConfig,
    problem,
    scene_sha256: str,
    objective_instance_sha256: str,
) -> NewtonConfig | None:
    """Authenticate the zero-step run eligible for alternate-only verification."""
    if reference.accepted or reference.newton_run is None:
        return None
    retry_config = dataclasses.replace(primary_config, step_relative_tolerance=0.0)
    retry_config.validate()
    record = _thaw_json(reference.deterministic_record)
    attempts = record.get("attempts")
    if (
        record.get("retry_policy")
        not in (
            _STALLED_RETRY_POLICY,
            _RESIDUAL_RECOVERY_RETRY_POLICY,
            _ALTERNATE_RECOVERY_RETRY_POLICY,
        )
        or record.get("selected_attempt") != 1
        or record.get("accepted") is not False
        or record.get("provenance_failures")
        or not isinstance(attempts, list)
        or len(attempts) != 2
    ):
        return None
    primary_attempt, retry_attempt = attempts
    if (
        primary_attempt.get("index") != 0
        or primary_attempt.get("role") != "primary"
        or retry_attempt.get("index") != 1
        or retry_attempt.get("role") != "step-relative-tolerance-zero"
    ):
        return None
    primary_record = primary_attempt.get("record")
    retry_record = retry_attempt.get("record")
    if not isinstance(primary_record, dict) or not isinstance(retry_record, dict):
        return None
    if primary_record.get("config") != dataclasses.asdict(primary_config):
        return None
    if retry_record.get("config") != dataclasses.asdict(retry_config):
        return None
    if record.get("config") != dataclasses.asdict(retry_config):
        return None
    for identity, expected in (
        ("scene_sha256", scene_sha256),
        ("objective_instance_sha256", objective_instance_sha256),
    ):
        if any(attempt.get(identity) != expected for attempt in (record, primary_record, retry_record)):
            return None
    position_sha256 = _array_digest(reference.positions)
    if record.get("position_sha256") != position_sha256 or retry_record.get("position_sha256") != position_sha256:
        return None
    run = reference.newton_run
    if (
        run.config != retry_config
        or run.reference_accepted
        or not run.result.converged
        or run.result.reason != "gradient"
        or not run.verification_converged
        or run.verification_reason != "gradient"
        or run.verification_displacement_relative > 1.0e-12
        or run.alternate_start_converged
        or run.alternate_start_reason not in ("stalled", "line_search", "max_iterations")
        or run.result_state_sha256 != position_sha256
        or _array_digest(run.result.x.detach().numpy()) != position_sha256
    ):
        return None
    if retry_record != _dense_reference_deterministic_record(run):
        return None
    if len(reference.failures) != 1 or not reference.failures[0].startswith("alternate-start gradient "):
        return None
    residual_limit = max(1.0e-10, 1.0e-10 * problem.residual_scale)
    if (
        run.result.final_gradient_norm > residual_limit
        or run.alternate_start_gradient_norm <= residual_limit
        or run.alternate_start_displacement_relative > 1.0e-9
    ):
        return None
    try:
        metrics = evaluate_common_state(problem, reference.positions)
    except (RuntimeError, ValueError):
        return None
    if metrics.gradient_norm > residual_limit or metrics.max_pin_error_m != 0.0 or metrics.inverted_tet_fraction != 0.0:
        return None
    return retry_config


def _recovery_provenance_failures(
    reference: _ReferenceStep,
    recovery: NewtonResidualReferenceRecovery,
    retry_config: NewtonConfig,
    problem,
    scene_sha256: str,
    objective_instance_sha256: str,
    displacement_scale: float,
) -> tuple[str, ...]:
    """Authenticate a three-start recovery before selecting its endpoint."""
    failures = []
    record = recovery.deterministic_record()
    expected_polish_config = NewtonResidualPolishConfig.from_newton_config(retry_config)
    if record.get("source_newton_config") != dataclasses.asdict(retry_config):
        failures.append("residual recovery source config does not match the zero-step Newton retry")
    if record.get("polish_config") != expected_polish_config.deterministic_record():
        failures.append("residual recovery polish config does not match the requested polish config")
    if recovery.source_newton_config != retry_config:
        failures.append("residual recovery object changed the zero-step Newton retry config")
    if recovery.polish_config != expected_polish_config:
        failures.append("residual recovery object changed the requested polish config")
    if reference.newton_run is None or recovery.source_run is not reference.newton_run:
        failures.append("residual recovery is not bound to the selected zero-step Newton run")
    if recovery.displacement_scale != displacement_scale or record.get("displacement_scale") != displacement_scale:
        failures.append("residual recovery changed the displacement normalization scale")
    expected_gradient_limit = max(
        retry_config.gradient_absolute_tolerance,
        retry_config.gradient_relative_tolerance * problem.residual_scale,
    )
    for role in ("canonical", "verification", "alternate"):
        result = getattr(recovery, role)
        if result.residual_scale != problem.residual_scale:
            failures.append(f"residual recovery {role} changed the problem residual scale")
        if result.gradient_limit != expected_gradient_limit:
            failures.append(f"residual recovery {role} changed the configured gradient limit")
    recomputed_displacements = {
        "verification_displacement_relative": float(
            np.linalg.norm(recovery.verification.x.detach().numpy() - recovery.canonical.x.detach().numpy())
        )
        / displacement_scale,
        "alternate_start_displacement_relative": float(
            np.linalg.norm(recovery.alternate.x.detach().numpy() - recovery.canonical.x.detach().numpy())
        )
        / displacement_scale,
    }
    for name, recomputed in recomputed_displacements.items():
        recorded = getattr(recovery, name)
        roundoff = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(recorded), abs(recomputed))
        if abs(recorded - recomputed) > roundoff or record.get(name) != recorded:
            failures.append(f"residual recovery {name} does not match its attempt positions")

    for identity, expected in (
        ("scene_sha256", scene_sha256),
        ("objective_instance_sha256", objective_instance_sha256),
    ):
        if expected is None or recovery.deterministic_record().get(identity) != expected:
            failures.append(f"residual recovery changed {identity}")
        if reference.deterministic_record.get(identity) != expected:
            failures.append(f"rejected Newton retry changed {identity}")

    rejected_sha256 = _array_digest(reference.positions)
    if recovery.rejected_state_sha256 != rejected_sha256 or record.get("rejected_state_sha256") != rejected_sha256:
        failures.append("residual recovery rejected-state hash does not match the zero-step Newton retry")
    if recovery.rejected_metrics.position_sha256 != rejected_sha256:
        failures.append("residual recovery rejected metrics belong to different positions")
    try:
        expected_rejected_metrics = evaluate_common_state(problem, reference.positions)
    except (RuntimeError, ValueError) as exc:
        failures.append(f"residual recovery rejected-state evaluation raised {type(exc).__name__}: {exc}")
    else:
        if recovery.rejected_metrics.as_dict() != expected_rejected_metrics.as_dict():
            failures.append("residual recovery rejected metrics do not match independent evaluation")
    attempt_records = record.get("attempts", {})
    for role in ("canonical", "verification", "alternate"):
        result = getattr(recovery, role)
        metrics = getattr(recovery, f"{role}_metrics")
        position_sha256 = _array_digest(result.x.detach().numpy())
        attempt_record = attempt_records.get(role, {})
        if attempt_record.get("position_sha256") != position_sha256:
            failures.append(f"residual recovery {role} position hash does not match its positions")
        if metrics is not None and metrics.position_sha256 != position_sha256:
            failures.append(f"residual recovery {role} metrics belong to different positions")
        try:
            expected_metrics = evaluate_common_state(problem, result.x)
        except (RuntimeError, ValueError) as exc:
            failures.append(f"residual recovery {role} evaluation raised {type(exc).__name__}: {exc}")
        else:
            if metrics is None or metrics.as_dict() != expected_metrics.as_dict():
                failures.append(f"residual recovery {role} metrics do not match independent evaluation")
            for name, recorded, expected in (
                ("objective", result.final_objective, expected_metrics.objective),
                ("gradient", result.final_gradient_norm, expected_metrics.gradient_norm),
                ("relative residual", result.final_relative_residual, expected_metrics.relative_residual),
            ):
                roundoff = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(recorded), abs(expected))
                if not math.isfinite(recorded) or abs(recorded - expected) > roundoff:
                    failures.append(f"residual recovery {role} final {name} does not match independent evaluation")

    digest_payload = dict(record)
    declared_digest = digest_payload.pop("recovery_sha256", None)
    if declared_digest != recovery.recovery_sha256 or _canonical_digest(digest_payload) != recovery.recovery_sha256:
        failures.append("residual recovery deterministic record hash is invalid")
    if recovery.reference_accepted:
        if record.get("reference_accepted") is not True or record.get("reference_failures") != []:
            failures.append("accepted residual recovery record does not declare acceptance")
        if any(
            metrics is None
            for metrics in (
                recovery.canonical_metrics,
                recovery.verification_metrics,
                recovery.alternate_metrics,
            )
        ):
            failures.append("accepted residual recovery omitted common-state metrics")
    return tuple(failures)


def _recovery_attempt_records(reference: _ReferenceStep, *, timing: bool) -> list[dict[str, object]]:
    """Copy the authenticated primary/zero-step records for a third attempt."""
    source = reference.timing_record if timing else reference.deterministic_record
    record = _thaw_json(source)
    attempts = record.get("attempts")
    if isinstance(attempts, list):
        return attempts
    return [{"index": 0, "role": "primary", "record": record}]


def _combine_reference_recovery(
    reference: _ReferenceStep,
    recovery: NewtonResidualReferenceRecovery,
    retry_config: NewtonConfig,
    provenance_failures: tuple[str, ...] = (),
) -> _ReferenceStep:
    """Bind ordinary and three-start-polish evidence into one selected record."""
    selected_failures = provenance_failures or recovery.reference_failures
    selected_accepted = recovery.reference_accepted and not provenance_failures
    result_record = recovery.canonical.deterministic_record()
    deterministic_record = {
        "method": "dense-cpu-newton-float64-with-strict-residual-polish",
        "config": dataclasses.asdict(retry_config),
        "scene_sha256": recovery.scene_sha256,
        "objective_instance_sha256": recovery.objective_instance_sha256,
        "accepted": selected_accepted,
        "failures": list(selected_failures),
        "native_converged": recovery.canonical.converged,
        "native_reason": recovery.canonical.reason,
        "accepted_iterations": recovery.canonical.accepted_iterations,
        "final_objective": recovery.canonical.final_objective,
        "final_gradient_norm": recovery.canonical.final_gradient_norm,
        "final_relative_residual": recovery.canonical.final_relative_residual,
        "verification_displacement_relative": recovery.verification_displacement_relative,
        "verification_converged": recovery.verification.converged,
        "verification_reason": recovery.verification.reason,
        "alternate_start_displacement_relative": recovery.alternate_start_displacement_relative,
        "alternate_start_converged": recovery.alternate.converged,
        "alternate_start_reason": recovery.alternate.reason,
        "alternate_start_gradient_norm": (
            None if recovery.alternate_metrics is None else recovery.alternate_metrics.gradient_norm
        ),
        "alternate_start_relative_residual": (
            None if recovery.alternate_metrics is None else recovery.alternate_metrics.relative_residual
        ),
        "work": result_record["work"],
        "trace": result_record["trace"],
        "position_sha256": _array_digest(recovery.canonical.x.detach().numpy()),
        "prerequisite_retry_policy": reference.deterministic_record.get("retry_policy"),
        "residual_polish_policy": _RESIDUAL_POLISH_POLICY,
        "selected_attempt": 2,
        "attempts": [
            *_recovery_attempt_records(reference, timing=False),
            {
                "index": 2,
                "role": "strict-residual-polish",
                "record": recovery.deterministic_record(),
            },
        ],
    }
    if provenance_failures:
        deterministic_record["provenance_failures"] = list(provenance_failures)
    timing_record = {
        "prerequisite_retry_policy": reference.timing_record.get("retry_policy"),
        "residual_polish_policy": _RESIDUAL_POLISH_POLICY,
        "selected_attempt": 2,
        "attempts": [
            *_recovery_attempt_records(reference, timing=True),
            {
                "index": 2,
                "role": "strict-residual-polish",
                "record": recovery.timing_record(),
            },
        ],
    }
    return _ReferenceStep(
        positions=recovery.canonical.x.detach().numpy(),
        accepted=selected_accepted,
        failures=selected_failures,
        deterministic_record=deterministic_record,
        timing_record=timing_record,
    )


def _combine_reference_recovery_exception(
    reference: _ReferenceStep,
    retry_config: NewtonConfig,
    exc: Exception,
) -> _ReferenceStep:
    """Preserve both ordinary attempts when residual recovery raises."""
    message = f"strict residual reference recovery raised {type(exc).__name__}: {exc}"
    deterministic_record = dict(_thaw_json(reference.deterministic_record))
    deterministic_record.update(
        {
            "accepted": False,
            "failures": [message],
            "prerequisite_retry_policy": reference.deterministic_record.get("retry_policy"),
            "residual_polish_policy": _RESIDUAL_POLISH_POLICY,
            "selected_attempt": 2,
            "attempts": [
                *_recovery_attempt_records(reference, timing=False),
                {
                    "index": 2,
                    "role": "strict-residual-polish",
                    "record": {
                        "method": _RESIDUAL_POLISH_POLICY,
                        "source_newton_config": dataclasses.asdict(retry_config),
                        "polish_config": NewtonResidualPolishConfig.from_newton_config(
                            retry_config
                        ).deterministic_record(),
                        "reference_accepted": False,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                },
            ],
        }
    )
    timing_record = {
        "prerequisite_retry_policy": reference.timing_record.get("retry_policy"),
        "residual_polish_policy": _RESIDUAL_POLISH_POLICY,
        "selected_attempt": 2,
        "attempts": [
            *_recovery_attempt_records(reference, timing=True),
            {
                "index": 2,
                "role": "strict-residual-polish",
                "record": {"unavailable_due_to_exception": True},
            },
        ],
    }
    return _ReferenceStep(
        positions=reference.positions,
        accepted=False,
        failures=(message,),
        deterministic_record=deterministic_record,
        timing_record=timing_record,
    )


def _alternate_verification_provenance_failures(
    reference: _ReferenceStep,
    verification: NewtonAlternateResidualVerification,
    retry_config: NewtonConfig,
    problem,
    scene_sha256: str,
    objective_instance_sha256: str,
    displacement_scale: float,
) -> tuple[str, ...]:
    """Independently authenticate alternate-only residual evidence."""
    failures = []
    record = verification.deterministic_record()
    expected_polish_config = NewtonResidualPolishConfig.from_newton_config(retry_config)
    if record.get("source_newton_config") != dataclasses.asdict(retry_config):
        failures.append("alternate residual verification source config does not match the zero-step retry")
    if record.get("polish_config") != expected_polish_config.deterministic_record():
        failures.append("alternate residual verification polish config does not match the requested config")
    if verification.source_run is not reference.newton_run:
        failures.append("alternate residual verification is not bound to the selected zero-step Newton run")
    for identity, expected in (
        ("scene_sha256", scene_sha256),
        ("objective_instance_sha256", objective_instance_sha256),
    ):
        if record.get(identity) != expected or reference.deterministic_record.get(identity) != expected:
            failures.append(f"alternate residual verification changed {identity}")

    representative_sha256 = _array_digest(reference.positions)
    if (
        verification.representative_state_sha256 != representative_sha256
        or record.get("representative_state_sha256") != representative_sha256
    ):
        failures.append("alternate residual verification changed the representative state hash")
    if verification.displacement_scale != displacement_scale or record.get("displacement_scale") != displacement_scale:
        failures.append("alternate residual verification changed the displacement normalization scale")
    source_alternate_record = record.get("ordinary_alternate", {})
    recomputed_source_alternate_displacement = (
        float(np.linalg.norm(verification.source_alternate.x.detach().numpy() - reference.positions))
        / displacement_scale
    )
    source_alternate_roundoff = (
        8.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(verification.source_alternate_displacement_relative),
            abs(recomputed_source_alternate_displacement),
        )
    )
    if (
        abs(verification.source_alternate_displacement_relative - recomputed_source_alternate_displacement)
        > source_alternate_roundoff
        or source_alternate_record.get("displacement_relative") != verification.source_alternate_displacement_relative
    ):
        failures.append("ordinary alternate displacement does not match its positions")
    recomputed_displacements = {
        "alternate_displacement_relative": float(
            np.linalg.norm(verification.alternate.x.detach().numpy() - reference.positions)
        )
        / displacement_scale,
        "repeat_displacement_relative": float(
            np.linalg.norm(verification.repeat.x.detach().numpy() - verification.alternate.x.detach().numpy())
        )
        / displacement_scale,
    }
    for name, recomputed in recomputed_displacements.items():
        recorded = getattr(verification, name)
        roundoff = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(recorded), abs(recomputed))
        if abs(recorded - recomputed) > roundoff or record.get(name) != recorded:
            failures.append(f"alternate residual verification {name} does not match its positions")

    expected_gradient_limit = max(
        retry_config.gradient_absolute_tolerance,
        retry_config.gradient_relative_tolerance * problem.residual_scale,
    )
    for role in ("alternate", "repeat"):
        result = getattr(verification, role)
        if result.residual_scale != problem.residual_scale:
            failures.append(f"alternate residual verification {role} changed the problem residual scale")
        if result.gradient_limit != expected_gradient_limit:
            failures.append(f"alternate residual verification {role} changed the configured gradient limit")

    expected_states = {
        "representative": reference.positions,
        "source_alternate": verification.source_alternate.x,
    }
    expected_states.update(
        {
            "alternate": verification.alternate.x,
            "repeat": verification.repeat.x,
        }
    )
    stored_metrics = {
        "representative": verification.representative_metrics,
        "source_alternate": verification.source_alternate_metrics,
    }
    stored_metrics.update(
        {
            "alternate": verification.alternate_metrics,
            "repeat": verification.repeat_metrics,
        }
    )
    for role, positions in expected_states.items():
        try:
            expected_metrics = evaluate_common_state(problem, positions)
        except (RuntimeError, ValueError) as exc:
            failures.append(f"alternate residual verification {role} evaluation raised {type(exc).__name__}: {exc}")
            continue
        metrics = stored_metrics[role]
        if metrics is None or metrics.as_dict() != expected_metrics.as_dict():
            failures.append(f"alternate residual verification {role} metrics do not match independent evaluation")
        if role in ("alternate", "repeat"):
            result = getattr(verification, role)
            for name, recorded, expected in (
                ("objective", result.final_objective, expected_metrics.objective),
                ("gradient", result.final_gradient_norm, expected_metrics.gradient_norm),
                ("relative residual", result.final_relative_residual, expected_metrics.relative_residual),
            ):
                roundoff = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(recorded), abs(expected))
                if not math.isfinite(recorded) or abs(recorded - expected) > roundoff:
                    failures.append(
                        f"alternate residual verification {role} final {name} does not match independent evaluation"
                    )

    attempt_records = record.get("attempts", {})
    for role in ("alternate", "repeat"):
        result = getattr(verification, role)
        position_sha256 = _array_digest(result.x.detach().numpy())
        attempt_record = attempt_records.get(role, {})
        if attempt_record.get("position_sha256") != position_sha256:
            failures.append(f"alternate residual verification {role} position hash is invalid")
    source_alternate_sha256 = _array_digest(verification.source_alternate.x.detach().numpy())
    if source_alternate_record.get("position_sha256") != source_alternate_sha256:
        failures.append("ordinary alternate position hash is invalid")
    if source_alternate_record.get("result") != _newton_result_deterministic_record(verification.source_alternate):
        failures.append("ordinary alternate deterministic result record is invalid")
    if source_alternate_record.get("metrics") != verification.source_alternate_metrics.as_dict():
        failures.append("ordinary alternate metrics record is invalid")
    if source_alternate_record.get("displacement_relative") != verification.source_alternate_displacement_relative:
        failures.append("ordinary alternate displacement record is invalid")
    source_run = verification.source_run
    if (
        verification.source_alternate.converged != source_run.alternate_start_converged
        or verification.source_alternate.reason != source_run.alternate_start_reason
        or verification.source_alternate_metrics.gradient_norm != source_run.alternate_start_gradient_norm
        or verification.source_alternate_metrics.relative_residual != source_run.alternate_start_relative_residual
    ):
        failures.append("ordinary alternate result does not reproduce the source Newton run summary")
    digest_payload = dict(record)
    declared_digest = digest_payload.pop("verification_sha256", None)
    if (
        declared_digest != verification.verification_sha256
        or _canonical_digest(digest_payload) != verification.verification_sha256
    ):
        failures.append("alternate residual verification deterministic record hash is invalid")
    if verification.reference_accepted:
        if record.get("reference_accepted") is not True or record.get("reference_failures") != []:
            failures.append("accepted alternate residual verification record does not declare acceptance")
        if verification.alternate_metrics is None or verification.repeat_metrics is None:
            failures.append("accepted alternate residual verification omitted common-state metrics")
    return tuple(failures)


def _combine_alternate_residual_verification(
    reference: _ReferenceStep,
    verification: NewtonAlternateResidualVerification,
    provenance_failures: tuple[str, ...] = (),
) -> _ReferenceStep:
    """Select the unchanged representative after alternate residual verification."""
    selected_failures = provenance_failures or verification.reference_failures
    selected_accepted = verification.reference_accepted and not provenance_failures
    deterministic_record = dict(_thaw_json(reference.deterministic_record))
    deterministic_record.update(
        {
            "method": "dense-cpu-newton-float64-with-alternate-residual-verification",
            "accepted": selected_accepted,
            "failures": list(selected_failures),
            "position_sha256": _array_digest(reference.positions),
            "prerequisite_retry_policy": reference.deterministic_record.get("retry_policy"),
            "alternate_residual_policy": _ALTERNATE_RESIDUAL_POLICY,
            "selected_attempt": 2,
            "selected_state": "zero-step-newton-representative",
            "attempts": [
                *_recovery_attempt_records(reference, timing=False),
                {
                    "index": 2,
                    "role": "alternate-start-residual-verification",
                    "record": verification.deterministic_record(),
                },
            ],
        }
    )
    if provenance_failures:
        deterministic_record["provenance_failures"] = list(provenance_failures)
    timing_record = {
        "prerequisite_retry_policy": reference.timing_record.get("retry_policy"),
        "alternate_residual_policy": _ALTERNATE_RESIDUAL_POLICY,
        "selected_attempt": 2,
        "attempts": [
            *_recovery_attempt_records(reference, timing=True),
            {
                "index": 2,
                "role": "alternate-start-residual-verification",
                "record": verification.timing_record(),
            },
        ],
    }
    return _ReferenceStep(
        positions=reference.positions,
        accepted=selected_accepted,
        failures=selected_failures,
        deterministic_record=deterministic_record,
        timing_record=timing_record,
    )


def _combine_alternate_residual_verification_exception(
    reference: _ReferenceStep,
    retry_config: NewtonConfig,
    exc: Exception,
) -> _ReferenceStep:
    """Preserve both ordinary attempts when alternate verification raises."""
    message = f"alternate residual verification raised {type(exc).__name__}: {exc}"
    deterministic_record = dict(_thaw_json(reference.deterministic_record))
    deterministic_record.update(
        {
            "accepted": False,
            "failures": [message],
            "prerequisite_retry_policy": reference.deterministic_record.get("retry_policy"),
            "alternate_residual_policy": _ALTERNATE_RESIDUAL_POLICY,
            "selected_attempt": 2,
            "attempts": [
                *_recovery_attempt_records(reference, timing=False),
                {
                    "index": 2,
                    "role": "alternate-start-residual-verification",
                    "record": {
                        "method": _ALTERNATE_RESIDUAL_POLICY,
                        "source_newton_config": dataclasses.asdict(retry_config),
                        "polish_config": NewtonResidualPolishConfig.from_newton_config(
                            retry_config
                        ).deterministic_record(),
                        "reference_accepted": False,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                },
            ],
        }
    )
    timing_record = {
        "prerequisite_retry_policy": reference.timing_record.get("retry_policy"),
        "alternate_residual_policy": _ALTERNATE_RESIDUAL_POLICY,
        "selected_attempt": 2,
        "attempts": [
            *_recovery_attempt_records(reference, timing=True),
            {
                "index": 2,
                "role": "alternate-start-residual-verification",
                "record": {"unavailable_due_to_exception": True},
            },
        ],
    }
    return _ReferenceStep(
        positions=reference.positions,
        accepted=False,
        failures=(message,),
        deterministic_record=deterministic_record,
        timing_record=timing_record,
    )


def _solve_dense_reference_with_recovery(
    scene: TetBenchmarkScene,
    problem,
    config: NewtonConfig,
) -> _ReferenceStep:
    """Run ordinary attempts first, then conditionally try strict residual polish."""
    reference = _solve_dense_reference_with_retry(scene, config)
    if reference.accepted:
        return reference
    reference = _run_residual_recovery_zero_step_prerequisite(scene, reference, config)
    if reference.accepted:
        return reference
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(common_objective_manifest(scene, problem)["objective_instance_sha256"])
    free_count = int(problem.free.numel())
    bbox_diagonal = float(np.linalg.norm(scene.rest_q.max(axis=0) - scene.rest_q.min(axis=0)))
    displacement_scale = max(math.sqrt(free_count) * bbox_diagonal, 1.0e-30)
    retry_config = _residual_polish_retry_config(
        reference,
        config,
        scene_sha256,
        objective_instance_sha256,
    )
    if retry_config is not None and reference.newton_run is not None:
        try:
            recovery = recover_newton_reference_with_residual_polish(
                scene,
                problem,
                reference.newton_run,
            )
            provenance_failures = _recovery_provenance_failures(
                reference,
                recovery,
                retry_config,
                problem,
                scene_sha256,
                objective_instance_sha256,
                displacement_scale,
            )
            return _combine_reference_recovery(reference, recovery, retry_config, provenance_failures)
        except Exception as exc:
            return _combine_reference_recovery_exception(reference, retry_config, exc)

    reference = _run_alternate_verification_zero_step_prerequisite(scene, problem, reference, config)
    if reference.accepted:
        return reference
    alternate_retry_config = _alternate_verification_retry_config(
        reference,
        config,
        problem,
        scene_sha256,
        objective_instance_sha256,
    )
    if alternate_retry_config is None or reference.newton_run is None:
        return reference
    try:
        verification = verify_newton_alternate_start_with_residual_polish(
            scene,
            problem,
            reference.newton_run,
        )
        provenance_failures = _alternate_verification_provenance_failures(
            reference,
            verification,
            alternate_retry_config,
            problem,
            scene_sha256,
            objective_instance_sha256,
            displacement_scale,
        )
        return _combine_alternate_residual_verification(reference, verification, provenance_failures)
    except Exception as exc:
        return _combine_alternate_residual_verification_exception(reference, alternate_retry_config, exc)


class PRSceneHistory:
    """Exact PR callback schedule and fail-closed dense-Newton chain runner."""

    def __init__(self, kind: str):
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}")
        self.kind = kind
        self._base_scene = self._build_base_scene(kind)
        rest = self._base_scene.rest_q.astype(np.float32)

        if kind == "stretch":
            self._driven = np.where(np.isclose(rest[:, 0], np.float32(0.5), rtol=0.0, atol=1.0e-6))[0]
            source_path = "newton/examples/vbd/example_soft_beam_stretch.py"
            compression_ratio = None
            schedule = {
                "formula": "stretch(f)=2 if f>=200 else 1+f/200",
                "ramp_frames": 200,
                "endpoint": 2.0,
                "callback": "overwrite right-face x once at each frame start",
            }
        elif kind == "twist":
            self._driven = np.where(np.isclose(rest[:, 2], np.float32(1.8), rtol=0.0, atol=1.0e-6))[0]
            source_path = "newton/examples/vbd/example_soft_beam_twist.py"
            compression_ratio = None
            schedule = {
                "formula": "angle(f)=2*pi if f>=200 else (f/200)*2*pi",
                "ramp_frames": 200,
                "endpoint_radians": 2.0 * np.pi,
                "callback": "overwrite top-face x/y once at each frame start",
            }
        else:
            self._driven = np.where(np.isclose(rest[:, 2], np.float32(1.3), rtol=0.0, atol=1.0e-6))[0]
            source_path = "newton/examples/vbd/example_soft_cube_compression.py"
            compression_ratio = 0.5 if kind == "compression-50" else 0.1
            schedule = {
                "formula": "ratio(f)=1-(f/149)*(1-r) for f<150",
                "compression_frames": 150,
                "release_frame": 150,
                "endpoint_ratio": compression_ratio,
                "callback": "overwrite top-face z before release; at frame 150 activate it without overwrite",
            }

        self.manifest = PRHistoryManifest(
            kind=kind,
            source_path=source_path,
            base_physical_sha256=_base_physical_digest(self._base_scene),
            topology_sha256=_topology_digest(self._base_scene),
            material_sha256=_material_digest(self._base_scene),
            compression_ratio=compression_ratio,
            schedule=schedule,
        )
        self._static_bundle = PRHistoryStaticBundle(
            manifest_sha256=self.manifest.manifest_sha256,
            base_physical_sha256=self.manifest.base_physical_sha256,
            topology_sha256=self.manifest.topology_sha256,
            material_sha256=self.manifest.material_sha256,
            rest_q=self._base_scene.rest_q,
            tet_indices=self._base_scene.tet_indices,
            tet_poses=self._base_scene.tet_poses,
            mass=self._base_scene.mass,
            tet_materials=self._base_scene.tet_materials,
            gravity=self._base_scene.gravity,
            external_force=self._base_scene.external_force,
        )
        initial_state = CommittedState(
            manifest_sha256=self.manifest.manifest_sha256,
            coordinate=AtomicCoordinate(0, 0),
            q=rest,
            qd=self._base_scene.velocity.astype(np.float32),
            particle_flags=self._base_scene.particle_flags,
        )
        self.initial_checkpoint = HistoryCheckpoint(
            manifest_sha256=self.manifest.manifest_sha256,
            state=initial_state,
            prior_transition_sha256=None,
            prefix_sha256=_root_prefix(self.manifest.manifest_sha256, initial_state.state_sha256),
        )

    @property
    def static_bundle(self) -> PRHistoryStaticBundle:
        """Canonical static arrays paired with every transition sample."""
        return self._static_bundle

    @staticmethod
    def _build_base_scene(kind: str) -> TetBenchmarkScene:
        if kind == "stretch":
            return build_stretch_scene(stretch_ratio=1.0, one_shot_diagnostic=True, dt=_DT)
        if kind == "twist":
            return build_twist_scene(twist_angle=0.0, one_shot_diagnostic=True, dt=_DT)
        return build_compression_scene(
            compression_ratio=1.0,
            one_shot_diagnostic=True,
            dt=_DT,
        )

    def frame_schedule(self, frame: int) -> FrameSchedule:
        """Return the exact float64 callback scalar used by the PR source."""
        if isinstance(frame, bool) or not isinstance(frame, numbers.Integral) or not 0 <= frame < _TOTAL_FRAMES:
            raise ValueError(f"frame must lie in [0, {_TOTAL_FRAMES})")
        frame = int(frame)
        if self.kind == "stretch":
            value = 2.0 if frame >= 200 else 1.0 + (frame / 200) * (2.0 - 1.0)
            return FrameSchedule(frame, "drive", "stretch_ratio", value, False)
        if self.kind == "twist":
            value = 2.0 * np.pi if frame >= 200 else (frame / 200) * (2.0 * np.pi)
            return FrameSchedule(frame, "drive", "twist_angle_rad", value, False)

        ratio = float(self.manifest.compression_ratio)
        if frame < 150:
            t = min(frame / max(150 - 1, 1), 1.0)
            value = 1.0 - t * (1.0 - ratio)
            return FrameSchedule(frame, "drive", "compression_ratio", value, False)
        if frame == 150:
            return FrameSchedule(frame, "release", None, None, True)
        return FrameSchedule(frame, "none", None, None, True)

    def _validate_state(self, state: CommittedState) -> None:
        if state.manifest_sha256 != self.manifest.manifest_sha256:
            raise ValueError("state belongs to a different history manifest")
        if state.q.shape != self._base_scene.rest_q.shape:
            raise ValueError("state topology does not match the history manifest")
        if not 0 <= state.coordinate.ordinal <= self.manifest.transition_count:
            raise ValueError("state coordinate lies outside the history")

    def apply_callback(self, state: CommittedState) -> AppliedAtomicState:
        """Apply the source callback at substep zero and return ``A_k``."""
        self._validate_state(state)
        if state.coordinate == self.manifest.end_coordinate:
            raise ValueError("the exclusive history endpoint has no callback")
        q = state.q.copy()
        flags = state.particle_flags.copy()
        callback_applied = state.coordinate.substep == 0
        action = "none"
        value_name = None
        value = None

        if callback_applied:
            schedule = self.frame_schedule(state.coordinate.frame)
            action = schedule.action
            value_name = schedule.value_name
            value = schedule.value
            if self.kind == "stretch":
                target_x = schedule.value * (10 * 0.05)
                q[self._driven, 0] = target_x
            elif self.kind == "twist":
                angle = float(schedule.value)
                cosine = np.cos(angle)
                sine = np.sin(angle)
                center = np.array([3 * 0.05 / 2.0, 3 * 0.05 / 2.0])
                top_rest = self._base_scene.rest_q.astype(np.float32)[self._driven]
                # Preserve the scalar operation order of the PR source rather
                # than relying on a vectorized expression's rounding choices.
                for local_index, particle_index in enumerate(self._driven):
                    rx = top_rest[local_index, 0] - center[0]
                    ry = top_rest[local_index, 1] - center[1]
                    q[particle_index, 0] = center[0] + cosine * rx - sine * ry
                    q[particle_index, 1] = center[1] + sine * rx + cosine * ry
            elif action == "drive":
                target_z = 1.0 + schedule.value * (6 * 0.05)
                q[self._driven, 2] = target_z
            elif action == "release":
                flags[self._driven] |= _ACTIVE_FLAG

        pinned = np.where(
            (self._base_scene.mass.astype(np.float32) == np.float32(0.0)) | ((flags & _ACTIVE_FLAG) == 0)
        )[0].astype(np.int64)
        return AppliedAtomicState(
            manifest_sha256=self.manifest.manifest_sha256,
            coordinate=state.coordinate,
            input_state_sha256=state.state_sha256,
            callback_applied=callback_applied,
            action=action,
            schedule_value_name=value_name,
            schedule_value=value,
            q=q,
            particle_flags=flags,
            pinned_indices=pinned,
            pin_targets=q[pinned],
        )

    def build_atomic_scene(self, state: CommittedState, applied: AppliedAtomicState) -> TetBenchmarkScene:
        """Build one immutable common-objective scene from ``C_k`` and ``A_k``."""
        self._validate_state(state)
        if applied.manifest_sha256 != self.manifest.manifest_sha256:
            raise ValueError("applied state belongs to a different history manifest")
        if applied.coordinate != state.coordinate or applied.input_state_sha256 != state.state_sha256:
            raise ValueError("applied state does not descend from the committed state")
        metadata = dict(self._base_scene.metadata)
        metadata.update(
            {
                "state_kind": "audited PR callback history atomic substep",
                "history_manifest_sha256": self.manifest.manifest_sha256,
                "history_state_sha256": state.state_sha256,
                "history_applied_sha256": applied.applied_sha256,
                "history_frame_index": state.coordinate.frame,
                "history_substep_index": state.coordinate.substep,
                "history_callback_action": applied.action,
                "history_callback_applied": applied.callback_applied,
            }
        )
        return dataclasses.replace(
            self._base_scene,
            name=(f"pr2901-{self.kind}-history-f{state.coordinate.frame:03d}-s{state.coordinate.substep}"),
            x_current=state.q,
            velocity=state.qd,
            particle_flags=applied.particle_flags,
            pinned_indices=applied.pinned_indices,
            pin_targets=applied.pin_targets,
            metadata=metadata,
        )

    def _commit_reference(
        self,
        state: CommittedState,
        applied: AppliedAtomicState,
        reference_positions: np.ndarray,
    ) -> CommittedState:
        positions = np.asarray(reference_positions)
        if positions.shape != state.q.shape or not np.isfinite(positions).all():
            raise ValueError("accepted reference positions do not match the committed state")
        q_next = np.array(positions, dtype=np.float32, order="C", copy=True)
        if not np.array_equal(q_next[applied.pinned_indices], applied.pin_targets):
            raise ValueError("accepted reference does not preserve the exact float32 pin targets")
        # Both operands and the scalar divisor are float32, which preserves
        # the VBD update_velocity kernel's subtraction-then-division order.
        qd_next = (q_next - applied.q) / np.float32(self.manifest.dt_seconds)
        qd_next = np.asarray(qd_next, dtype=np.float32)
        if applied.pinned_indices.size:
            qd_next[applied.pinned_indices] = np.float32(0.0)
        return CommittedState(
            manifest_sha256=self.manifest.manifest_sha256,
            coordinate=state.coordinate.next(),
            q=q_next,
            qd=qd_next,
            particle_flags=applied.particle_flags,
        )

    def generate(
        self,
        *,
        start: AtomicCoordinate | None = None,
        stop: AtomicCoordinate | None = None,
        checkpoint: HistoryCheckpoint | None = None,
        prior_chain: PRHistoryChain | None = None,
        max_transitions: int = 8,
        newton_config: NewtonConfig | None = None,
    ) -> PRHistoryChain:
        """Generate a bounded accepted range, stopping on reference rejection.

        ``stop`` is exclusive.  Omitting it requests exactly one transition.
        Starting after ``(0, 0)`` requires both a checkpoint and the verified
        prior chain from which it was selected.  The returned suffix binds
        that proof by ``prior_chain_sha256``. Raising ``max_transitions`` is
        an explicit opt-in to a longer dense solve.
        """
        if (
            isinstance(max_transitions, bool)
            or not isinstance(max_transitions, numbers.Integral)
            or max_transitions < 1
        ):
            raise ValueError("max_transitions must be a positive integer")
        initial = self.initial_checkpoint if checkpoint is None else checkpoint
        if initial.manifest_sha256 != self.manifest.manifest_sha256:
            raise ValueError("checkpoint belongs to a different history manifest")
        self._validate_state(initial.state)
        if initial.state.coordinate == AtomicCoordinate(0, 0):
            if initial.checkpoint_sha256 != self.initial_checkpoint.checkpoint_sha256:
                raise ValueError("root checkpoint does not match the canonical initial history state")
            if prior_chain is not None:
                raise ValueError("the canonical root checkpoint must not declare a prior chain")
        else:
            if prior_chain is None:
                raise ValueError("a non-root checkpoint requires a verified prior-chain proof")
            prior_chain.verify()
            if prior_chain.manifest.manifest_sha256 != self.manifest.manifest_sha256:
                raise ValueError("prior-chain proof belongs to a different history manifest")
            expected_checkpoint = prior_chain.checkpoint_at(initial.state.coordinate)
            if expected_checkpoint.checkpoint_sha256 != initial.checkpoint_sha256:
                raise ValueError("checkpoint does not match the verified prior-chain proof")
        requested_start = initial.state.coordinate if start is None else start
        if requested_start != initial.state.coordinate:
            raise ValueError("start must equal the selected checkpoint coordinate")
        if checkpoint is None and requested_start != AtomicCoordinate(0, 0):
            raise ValueError("a nonzero start requires a verified checkpoint")
        requested_stop = requested_start.next() if stop is None else stop
        if requested_stop.ordinal <= requested_start.ordinal:
            raise ValueError("stop must follow start")
        if requested_stop.ordinal > self.manifest.transition_count:
            raise ValueError("stop lies beyond the 400-frame PR history")
        count = requested_stop.ordinal - requested_start.ordinal
        if count > int(max_transitions):
            raise ValueError(
                f"requested {count} transitions exceeds max_transitions={max_transitions}; increase the cap explicitly"
            )

        config = _default_newton_config() if newton_config is None else newton_config
        config.validate()
        current_checkpoint = initial
        transitions: list[HistoryTransition] = []
        timings: list[TransitionTiming] = []
        failure = None

        while current_checkpoint.state.coordinate.ordinal < requested_stop.ordinal:
            state = current_checkpoint.state
            applied = self.apply_callback(state)
            scene = self.build_atomic_scene(state, applied)
            problem = build_common_problem(scene)
            objective = common_objective_manifest(scene, problem)
            try:
                reference = _solve_dense_reference_with_recovery(scene, problem, config)
            except Exception as exc:
                message = f"dense reference raised {type(exc).__name__}: {exc}"
                reference = _ReferenceStep(
                    positions=scene.vbd_inertial_target,
                    accepted=False,
                    failures=(message,),
                    deterministic_record={
                        "method": "dense-cpu-newton-float64",
                        "accepted": False,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                    timing_record={"unavailable_due_to_exception": True},
                )
            timings.append(
                TransitionTiming(
                    coordinate=state.coordinate,
                    accepted=reference.accepted,
                    values=reference.timing_record,
                )
            )
            if not reference.accepted:
                failures = reference.failures or ("dense reference rejected without a reason",)
                failure = FailedReference(
                    manifest_sha256=self.manifest.manifest_sha256,
                    coordinate=state.coordinate,
                    input_state_sha256=state.state_sha256,
                    input_prefix_sha256=current_checkpoint.prefix_sha256,
                    applied_record=applied.as_dict(),
                    scene_sha256=str(scene.manifest()["scene_sha256"]),
                    objective_instance_sha256=str(objective["objective_instance_sha256"]),
                    reference_record=reference.deterministic_record,
                    failures=failures,
                )
                break

            output_state = self._commit_reference(state, applied, reference.positions)
            transition = HistoryTransition(
                manifest_sha256=self.manifest.manifest_sha256,
                coordinate=state.coordinate,
                next_coordinate=state.coordinate.next(),
                input_state_sha256=state.state_sha256,
                input_prefix_sha256=current_checkpoint.prefix_sha256,
                input_state=state,
                applied_state=applied,
                applied_record=applied.as_dict(),
                scene_sha256=str(scene.manifest()["scene_sha256"]),
                objective_instance_sha256=str(objective["objective_instance_sha256"]),
                dt_seconds=self.manifest.dt_seconds,
                topology_sha256=self.manifest.topology_sha256,
                material_sha256=self.manifest.material_sha256,
                inertial_target=scene.vbd_inertial_target,
                reference_record=reference.deterministic_record,
                reference_positions=reference.positions,
                output_state=output_state,
            )
            transitions.append(transition)
            current_checkpoint = HistoryCheckpoint(
                manifest_sha256=self.manifest.manifest_sha256,
                state=output_state,
                prior_transition_sha256=transition.transition_sha256,
                prefix_sha256=_advance_prefix(current_checkpoint.prefix_sha256, transition.transition_sha256),
            )

        if failure is not None:
            termination = "failed_reference"
        elif current_checkpoint.state.coordinate == self.manifest.end_coordinate:
            termination = "history_complete"
        else:
            termination = "range_complete"
        return PRHistoryChain(
            manifest=self.manifest,
            initial_checkpoint=initial,
            transitions=tuple(transitions),
            timings=tuple(timings),
            final_checkpoint=current_checkpoint,
            termination=termination,
            failed_reference=failure,
            prior_chain_sha256=None if prior_chain is None else prior_chain.chain_sha256,
        )


def create_pr_scene_history(kind: str) -> PRSceneHistory:
    """Create one exact stretch, twist, or compression history definition."""
    return PRSceneHistory(kind)
