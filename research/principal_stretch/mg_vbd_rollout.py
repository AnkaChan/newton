# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""History-preserving rollout adapter for the captured MG-VBD research lane.

The one-step benchmark deliberately freezes and authenticates one pristine
physical input.  Qualitative trajectories need different semantics: the
accepted endpoint of step ``n`` must become the physical input of step
``n + 1`` while topology, material data, and the captured solve schedule stay
fixed.  This module supplies that state machine without weakening or changing
the immutable one-step API in :mod:`captured_graph_vbd`.

The generic :class:`MGVBDRollout` is device independent.  Its backend contract
is small enough to test on CPU, while :class:`MGVBDRolloutCapturedBackend`
adapts one already-captured CUDA solver by temporarily updating only its
dynamic input arrays.  Those arrays are restored after every replay, including
exceptional exits, so the wrapped one-step solver retains its original
validation contract.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import numbers

import numpy as np
import warp as wp

from .captured_graph_vbd import REASON_NAMES, CapturedDirectGraphVBD, _lookup_workspace_owners
from .correction_graph_vbd import DirectGraphVBDConfig
from .solver_benchmark import TetBenchmarkScene, _vbd_inertial_target

CONTRACT_ID = "history-preserving-mg-vbd-rollout-v1"


def _immutable_array(value: object, dtype: np.dtype, *, name: str) -> np.ndarray:
    """Return a finite, C-contiguous array backed by immutable bytes."""
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    if owned.dtype.kind in "fc" and not np.isfinite(owned).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _float32_promoted(value: object, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    """Canonicalize one SolverVBD-facing array through float32."""
    array = _immutable_array(value, np.float32, name=name)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return _immutable_array(array, np.float64, name=name)


def _canonical_dt(value: object) -> float:
    """Canonicalize one positive timestep through SolverVBD float32."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("dt must be a real scalar")
    result = float(np.float32(value))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("dt must be finite and positive")
    return result


def _published_velocity(
    positions: np.ndarray,
    x_current: np.ndarray,
    pinned_indices: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reproduce the captured endpoint's float32 BDF1 publication."""
    velocity = ((positions - x_current) * np.float64(1.0 / dt)).astype(np.float32)
    velocity[pinned_indices] = np.float32(0.0)
    return _immutable_array(velocity, np.float64, name="published velocity")


@dataclasses.dataclass(frozen=True, eq=False)
class MGVBDRolloutStepInput:
    """Immutable physical input supplied to one reusable MG-VBD solve."""

    step_index: int
    dt: float
    positions: np.ndarray
    velocities: np.ndarray
    external_force: np.ndarray
    inertial_target: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative built-in int")
        dt = _canonical_dt(self.dt)
        if type(self.dt) is not float or self.dt != dt:
            raise ValueError("dt must be a canonical built-in float32-promoted value")

        positions = _immutable_array(self.positions, np.float64, name="positions")
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("positions must have shape (V, 3)")
        shape = positions.shape
        arrays = {
            "velocities": _immutable_array(self.velocities, np.float64, name="velocities"),
            "external_force": _immutable_array(self.external_force, np.float64, name="external_force"),
            "inertial_target": _immutable_array(self.inertial_target, np.float64, name="inertial_target"),
        }
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("rollout particle arrays must have matching shape (V, 3)")
        pinned = _immutable_array(self.pinned_indices, np.int64, name="pinned_indices")
        targets = _immutable_array(self.pin_targets, np.float64, name="pin_targets")
        if pinned.ndim != 1 or targets.shape != (pinned.size, 3):
            raise ValueError("pin_targets must have shape (P, 3) for one-dimensional pinned_indices")
        if pinned.size and (
            pinned[0] < 0 or pinned[-1] >= positions.shape[0] or not np.array_equal(pinned, np.unique(pinned))
        ):
            raise ValueError("pinned_indices must be sorted, unique, and in range")
        if pinned.size and not np.array_equal(positions[pinned], targets):
            raise ValueError("rollout input positions must match pin targets exactly")
        if pinned.size and np.any(arrays["velocities"][pinned] != 0.0):
            raise ValueError("rollout input pinned velocities must be exactly zero")
        if pinned.size and not np.array_equal(arrays["inertial_target"][pinned], targets):
            raise ValueError("rollout inertial target must match pin targets exactly")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", arrays["velocities"])
        object.__setattr__(self, "external_force", arrays["external_force"])
        object.__setattr__(self, "inertial_target", arrays["inertial_target"])
        object.__setattr__(self, "pinned_indices", pinned)
        object.__setattr__(self, "pin_targets", targets)


@dataclasses.dataclass(frozen=True, eq=False)
class MGVBDRolloutStepEndpoint:
    """One backend endpoint before the rollout advances persistent history."""

    positions: np.ndarray
    velocities: np.ndarray
    accepted: tuple[bool, ...] = ()
    reasons: tuple[str, ...] = ()
    graph_replay: bool = False

    def __post_init__(self) -> None:
        positions = _immutable_array(self.positions, np.float64, name="endpoint positions")
        velocities = _immutable_array(self.velocities, np.float64, name="endpoint velocities")
        if positions.ndim != 2 or positions.shape[1:] != (3,) or velocities.shape != positions.shape:
            raise ValueError("endpoint positions and velocities must have matching shape (V, 3)")
        if type(self.accepted) is not tuple or any(type(value) is not bool for value in self.accepted):
            raise ValueError("accepted must be an exact tuple of built-in bool values")
        if type(self.reasons) is not tuple or any(type(value) is not str for value in self.reasons):
            raise ValueError("reasons must be an exact tuple of built-in strings")
        if len(self.accepted) != len(self.reasons):
            raise ValueError("accepted and reasons must have matching lengths")
        if type(self.graph_replay) is not bool:
            raise ValueError("graph_replay must be a built-in bool")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)


@dataclasses.dataclass(frozen=True, eq=False)
class MGVBDRolloutState:
    """Immutable snapshot of the persistent state after one rollout boundary."""

    step_index: int
    time_seconds: float
    positions: np.ndarray
    velocities: np.ndarray
    external_force: np.ndarray
    inertial_target: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray
    accepted: tuple[bool, ...]
    reasons: tuple[str, ...]
    graph_replay: bool
    contract_id: str = CONTRACT_ID

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative built-in int")
        if type(self.time_seconds) is not float or not math.isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("time_seconds must be a finite non-negative built-in float")
        if type(self.contract_id) is not str or self.contract_id != CONTRACT_ID:
            raise ValueError("rollout state contract is not canonical")
        if type(self.accepted) is not tuple or any(type(value) is not bool for value in self.accepted):
            raise ValueError("accepted must be an exact tuple of built-in bool values")
        if type(self.reasons) is not tuple or any(type(value) is not str for value in self.reasons):
            raise ValueError("reasons must be an exact tuple of built-in strings")
        if len(self.accepted) != len(self.reasons):
            raise ValueError("accepted and reasons must have matching lengths")
        if type(self.graph_replay) is not bool:
            raise ValueError("graph_replay must be a built-in bool")

        positions = _immutable_array(self.positions, np.float64, name="state positions")
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("state positions must have shape (V, 3)")
        arrays = {
            "velocities": _immutable_array(self.velocities, np.float64, name="state velocities"),
            "external_force": _immutable_array(self.external_force, np.float64, name="state external_force"),
            "inertial_target": _immutable_array(self.inertial_target, np.float64, name="state inertial_target"),
        }
        if any(array.shape != positions.shape for array in arrays.values()):
            raise ValueError("state particle arrays must have matching shape (V, 3)")
        pinned = _immutable_array(self.pinned_indices, np.int64, name="state pinned_indices")
        targets = _immutable_array(self.pin_targets, np.float64, name="state pin_targets")
        if targets.shape != (pinned.size, 3):
            raise ValueError("state pin_targets must have shape (P, 3)")
        if pinned.size and not np.array_equal(positions[pinned], targets):
            raise ValueError("state positions must preserve pins exactly")
        if pinned.size and np.any(arrays["velocities"][pinned] != 0.0):
            raise ValueError("state pinned velocities must be exactly zero")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", arrays["velocities"])
        object.__setattr__(self, "external_force", arrays["external_force"])
        object.__setattr__(self, "inertial_target", arrays["inertial_target"])
        object.__setattr__(self, "pinned_indices", pinned)
        object.__setattr__(self, "pin_targets", targets)


class MGVBDRolloutBackend(abc.ABC):
    """Backend contract for one static-resource MG-VBD solve."""

    @property
    @abc.abstractmethod
    def scene(self) -> TetBenchmarkScene:
        """Canonical construction scene whose static data the backend owns."""

    @property
    @abc.abstractmethod
    def resource_identity(self) -> tuple[object, ...]:
        """Stable owner identity for the hierarchy and executable schedule."""

    @abc.abstractmethod
    def solve_step(self, step_input: MGVBDRolloutStepInput) -> MGVBDRolloutStepEndpoint:
        """Solve one dynamic input without replacing static resources."""


class MGVBDRollout:
    """Persistent history state machine around one reusable MG-VBD backend."""

    def __init__(self, scene: TetBenchmarkScene, backend: MGVBDRolloutBackend):
        if type(scene) is not TetBenchmarkScene:
            raise TypeError("scene must be an exact TetBenchmarkScene")
        if not isinstance(backend, MGVBDRolloutBackend):
            raise TypeError("backend must implement MGVBDRolloutBackend")
        if backend.scene is not scene:
            raise ValueError("backend and rollout must own the exact same construction scene")
        resource_identity = backend.resource_identity
        if type(resource_identity) is not tuple or not resource_identity:
            raise ValueError("backend resource_identity must be one non-empty exact tuple")

        self._scene = scene
        self._backend = backend
        self._resource_identity = resource_identity
        self._initial_positions = _float32_promoted(scene.x_current, (scene.n_vertices, 3), name="initial positions")
        self._initial_velocities = _float32_promoted(scene.velocity, (scene.n_vertices, 3), name="initial velocities")
        self._initial_force = _float32_promoted(
            scene.external_force,
            (scene.n_vertices, 3),
            name="initial external force",
        )
        self._initial_pin_targets = _float32_promoted(
            scene.pin_targets,
            (scene.pinned_indices.size, 3),
            name="initial pin targets",
        )
        self._pinned_indices = _immutable_array(scene.pinned_indices, np.int64, name="pinned_indices")
        self._positions = self._initial_positions
        self._velocities = self._initial_velocities
        self._external_force = self._initial_force
        self._pin_targets = self._initial_pin_targets
        self._inertial_target = _immutable_array(scene.vbd_inertial_target, np.float64, name="inertial target")
        self._step_index = 0
        self._time_seconds = 0.0
        self._accepted: tuple[bool, ...] = ()
        self._reasons: tuple[str, ...] = ()
        self._graph_replay = False
        self._enforce_pins()
        self._refresh_inertial_target(scene.dt)

    @property
    def scene(self) -> TetBenchmarkScene:
        """Canonical construction scene retained by the rollout."""
        return self._scene

    @property
    def resource_identity(self) -> tuple[object, ...]:
        """Static backend resource identity frozen at construction."""
        return self._resource_identity

    @property
    def state(self) -> MGVBDRolloutState:
        """Return an immutable snapshot of current persistent history."""
        return MGVBDRolloutState(
            step_index=self._step_index,
            time_seconds=float(self._time_seconds),
            positions=self._positions,
            velocities=self._velocities,
            external_force=self._external_force,
            inertial_target=self._inertial_target,
            pinned_indices=self._pinned_indices,
            pin_targets=self._pin_targets,
            accepted=self._accepted,
            reasons=self._reasons,
            graph_replay=self._graph_replay,
        )

    def _require_static_resources(self) -> None:
        """Reject backend recapture or hierarchy replacement mid-rollout."""
        identity = self._backend.resource_identity
        if type(identity) is not tuple or identity != self._resource_identity:
            raise RuntimeError("MG-VBD rollout backend resources changed after construction")

    def _enforce_pins(self) -> None:
        """Project persistent kinematic state to its exact current targets."""
        positions = np.array(self._positions, dtype=np.float64, order="C", copy=True)
        velocities = np.array(self._velocities, dtype=np.float64, order="C", copy=True)
        positions[self._pinned_indices] = self._pin_targets
        velocities[self._pinned_indices] = 0.0
        self._positions = _immutable_array(positions, np.float64, name="positions")
        self._velocities = _immutable_array(velocities, np.float64, name="velocities")

    def _refresh_inertial_target(self, dt: float) -> None:
        """Refresh the exact float32 SolverVBD inertial predictor."""
        self._inertial_target = _immutable_array(
            _vbd_inertial_target(
                self._positions,
                self._velocities,
                self._scene.gravity,
                self._external_force,
                self._scene.particle_inv_mass,
                self._pinned_indices,
                self._pin_targets,
                dt,
            ),
            np.float64,
            name="inertial target",
        )

    def set_external_force(self, external_force: np.ndarray) -> None:
        """Set the persistent per-particle external force [N]."""
        self._external_force = _float32_promoted(
            external_force,
            (self._scene.n_vertices, 3),
            name="external_force",
        )
        self._refresh_inertial_target(self._scene.dt)

    def set_pin_targets(self, pin_targets: np.ndarray) -> None:
        """Set targets [m] for the fixed construction-time pin set."""
        self._pin_targets = _float32_promoted(
            pin_targets,
            (self._pinned_indices.size, 3),
            name="pin_targets",
        )
        self._enforce_pins()
        self._refresh_inertial_target(self._scene.dt)

    def reset(self) -> MGVBDRolloutState:
        """Restore the exact initial state, loads, pins, and simulation time."""
        self._require_static_resources()
        self._positions = self._initial_positions
        self._velocities = self._initial_velocities
        self._external_force = self._initial_force
        self._pin_targets = self._initial_pin_targets
        self._step_index = 0
        self._time_seconds = 0.0
        self._accepted = ()
        self._reasons = ()
        self._graph_replay = False
        self._enforce_pins()
        self._refresh_inertial_target(self._scene.dt)
        return self.state

    def step(self, dt: float) -> MGVBDRolloutState:
        """Advance one history-preserving step with the captured timestep [s]."""
        canonical_dt = _canonical_dt(dt)
        if canonical_dt != self._scene.dt:
            raise ValueError(f"captured rollout timestep is fixed at {self._scene.dt!r}; got {canonical_dt!r}")
        self._require_static_resources()
        self._enforce_pins()
        self._refresh_inertial_target(canonical_dt)
        step_input = MGVBDRolloutStepInput(
            step_index=self._step_index,
            dt=canonical_dt,
            positions=self._positions,
            velocities=self._velocities,
            external_force=self._external_force,
            inertial_target=self._inertial_target,
            pinned_indices=self._pinned_indices,
            pin_targets=self._pin_targets,
        )
        endpoint = self._backend.solve_step(step_input)
        if type(endpoint) is not MGVBDRolloutStepEndpoint:
            raise TypeError("backend must return an exact MGVBDRolloutStepEndpoint")
        self._require_static_resources()

        shape = (self._scene.n_vertices, 3)
        positions = _float32_promoted(endpoint.positions, shape, name="endpoint positions")
        velocities = _float32_promoted(endpoint.velocities, shape, name="endpoint velocities")
        if not np.array_equal(positions, endpoint.positions):
            raise ValueError("backend endpoint positions are not exact float32-published values")
        if not np.array_equal(velocities, endpoint.velocities):
            raise ValueError("backend endpoint velocities are not exact float32-published values")
        if self._pinned_indices.size and not np.array_equal(positions[self._pinned_indices], self._pin_targets):
            raise ValueError("backend endpoint changed an exact pin")
        expected_velocity = _published_velocity(
            positions,
            self._positions,
            self._pinned_indices,
            canonical_dt,
        )
        if not np.array_equal(velocities, expected_velocity):
            raise ValueError("backend endpoint velocity is not the exact BDF1 update from x_n")

        # Commit only after all endpoint and static-resource checks pass.
        self._positions = positions
        self._velocities = velocities
        self._step_index += 1
        self._time_seconds = float(np.float64(self._time_seconds) + np.float64(canonical_dt))
        self._accepted = endpoint.accepted
        self._reasons = endpoint.reasons
        self._graph_replay = endpoint.graph_replay
        return self.state


class MGVBDRolloutCapturedBackend(MGVBDRolloutBackend):
    """Transactional dynamic-input adapter for one captured CUDA solver.

    Construction performs no capture.  Pass an already captured
    :class:`CapturedDirectGraphVBD`, or use :meth:`build` when GPU ownership has
    already been established by the caller.
    """

    def __init__(self, solver: object):
        if type(solver) is not CapturedDirectGraphVBD:
            raise TypeError("solver must be an exact CapturedDirectGraphVBD")
        owner_binding = _lookup_workspace_owners(solver)
        solver._validate_persistent_sources(owner_binding=owner_binding)
        if owner_binding.capture is None:
            raise RuntimeError("capture_graphs() must complete before constructing a rollout backend")

        self._solver = solver
        self._owner_binding = owner_binding
        self._scene = solver.scene
        self._resource_identity = self._live_resource_identity()
        public = owner_binding.public_vbd
        direct = owner_binding.direct
        operator = owner_binding.operator
        self._restore_arrays = (
            (public.pristine_input.particle_q, np.asarray(public.pristine_input.particle_q.numpy())),
            (public.pristine_input.particle_qd, np.asarray(public.pristine_input.particle_qd.numpy())),
            (public.pristine_input.particle_f, np.asarray(public.pristine_input.particle_f.numpy())),
            (operator.inertial_target, np.asarray(operator.inertial_target.numpy())),
            (direct.canonical_positions, np.asarray(direct.canonical_positions.numpy())),
            (direct.x_current, np.asarray(direct.x_current.numpy())),
        )
        self._canonical_positions = np.array(self._restore_arrays[4][1], dtype=np.float64, copy=True)
        self._in_solve = False

    @classmethod
    def build(
        cls,
        scene: TetBenchmarkScene,
        *,
        device: str = "cuda:0",
        config: DirectGraphVBDConfig | None = None,
        tile_solve: bool = False,
        warmup_replays: int = 1,
    ) -> MGVBDRolloutCapturedBackend:
        """Construct and capture one solver for a fixed rollout scene."""
        solver = CapturedDirectGraphVBD(
            scene,
            device=device,
            config=config,
            tile_solve=tile_solve,
        )
        solver.capture_graphs(warmup_replays=warmup_replays)
        return cls(solver)

    @property
    def scene(self) -> TetBenchmarkScene:
        """Canonical construction scene whose static data is captured."""
        return self._scene

    def _live_resource_identity(self) -> tuple[object, ...]:
        """Return the exact solver, hierarchy, graph, and capture generation."""
        owner_binding = _lookup_workspace_owners(self._solver)
        self._solver._validate_workspace_owner_bindings(owner_binding)
        capture = owner_binding.capture
        if capture is None:
            raise RuntimeError("captured rollout resource identity requires a live captured graph")
        return (
            id(self._solver),
            id(self._solver.hierarchy),
            id(self._solver.device_hierarchy),
            id(capture.graph),
            capture.generation,
        )

    @property
    def resource_identity(self) -> tuple[object, ...]:
        """Stable captured resource owner identity."""
        return self._live_resource_identity()

    def _assign_dynamic_inputs(self, step_input: MGVBDRolloutStepInput) -> None:
        """Assign only arrays whose pointers are intentionally graph-persistent."""
        public = self._owner_binding.public_vbd
        direct = self._owner_binding.direct
        operator = self._owner_binding.operator
        canonical_positions = self._canonical_positions.copy()
        canonical_positions[step_input.pinned_indices] = step_input.pin_targets
        public.pristine_input.particle_q.assign(np.asarray(step_input.positions, dtype=np.float32))
        public.pristine_input.particle_qd.assign(np.asarray(step_input.velocities, dtype=np.float32))
        public.pristine_input.particle_f.assign(np.asarray(step_input.external_force, dtype=np.float32))
        operator.inertial_target.assign(np.asarray(step_input.inertial_target, dtype=np.float64))
        direct.canonical_positions.assign(canonical_positions)
        direct.x_current.assign(np.asarray(step_input.positions, dtype=np.float64))

    def _restore_one_step_inputs(self) -> None:
        """Restore the one-step solver's exact construction-time source values."""
        for destination, source in self._restore_arrays:
            destination.assign(source)

    def solve_step(self, step_input: MGVBDRolloutStepInput) -> MGVBDRolloutStepEndpoint:
        """Replay one graph with dynamic history, then restore benchmark inputs."""
        if type(step_input) is not MGVBDRolloutStepInput:
            raise TypeError("step_input must be an exact MGVBDRolloutStepInput")
        if step_input.dt != self._scene.dt:
            raise ValueError("step_input dt differs from the captured solver timestep")
        if not np.array_equal(step_input.pinned_indices, self._scene.pinned_indices):
            raise ValueError("step_input changed the construction-time pin set")
        if self._in_solve:
            raise RuntimeError("captured rollout backend does not permit reentrant solves")
        if self.resource_identity != self._resource_identity:
            raise RuntimeError("captured rollout hierarchy or graph owner changed")

        self._solver._validate_persistent_sources(owner_binding=self._owner_binding)
        graph, replay_stream = self._solver._bound_graph_launch_owners(
            self._owner_binding,
            comparator=False,
        )
        self._in_solve = True
        try:
            with wp.ScopedStream(replay_stream, sync_enter=False):
                self._assign_dynamic_inputs(step_input)
                wp.capture_launch(graph, stream=replay_stream)
            positions = np.asarray(self._owner_binding.direct.final_positions.numpy(), dtype=np.float32).astype(
                np.float64
            )
            velocities = np.asarray(self._owner_binding.direct.final_velocities.numpy(), dtype=np.float32).astype(
                np.float64
            )
            accepted = tuple(bool(value) for value in self._owner_binding.direct.accepted.numpy())
            reason_codes = tuple(int(value) for value in self._owner_binding.direct.reasons.numpy())
            if any(code < 0 or code >= len(REASON_NAMES) for code in reason_codes):
                raise RuntimeError("captured rollout returned an invalid gate reason")
            reasons = tuple(REASON_NAMES[code] for code in reason_codes)
            return MGVBDRolloutStepEndpoint(
                positions=positions,
                velocities=velocities,
                accepted=accepted,
                reasons=reasons,
                graph_replay=True,
            )
        finally:
            try:
                with wp.ScopedStream(replay_stream, sync_enter=False):
                    self._restore_one_step_inputs()
                wp.synchronize_stream(replay_stream)
            finally:
                self._in_solve = False
