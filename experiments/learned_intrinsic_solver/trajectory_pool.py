# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental deterministic scheduling of mixed, independently sampled trajectories.

The caller performs one learned inner iteration on each returned record, finishes
backward, and replaces its candidate and optimizer history before returning the batch.
CPU preparation runs in bounded worker threads; the pool itself is owned by the
calling thread. Callbacks must protect any shared registries they mutate.
"""

from __future__ import annotations

import copy
import random
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import torch  # noqa: TID253

__all__ = ["ActiveTrajectoryPool", "TrajectoryRecord"]


@dataclass
class TrajectoryRecord:
    """Experimental independent physical trajectory with immutable sampled budgets."""

    id: int
    """Unique trajectory identity within this pool, including retired trajectories."""
    seed: int
    """Unique reset seed, assigned monotonically within this pool."""
    iteration_budget: int
    """Number of learned inner iterations per physical timestep."""
    step_budget: int
    """Number of physical timesteps before resetting the trajectory."""
    inner_iteration: int = 0
    """Zero-based next inner iteration within the current physical timestep."""
    physical_step: int = 0
    """Zero-based current physical timestep."""
    payload: dict[str, Any] = field(default_factory=dict)
    """Unbatched training inputs; empty until initial preparation completes."""


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _counts(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        _positive(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain distinct counts")
    return result


def _payload(value: Any, *, checkpoint: bool = False) -> Any:
    """Detach iteration boundaries and optionally isolate checkpoint storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if checkpoint else value.detach()
    if isinstance(value, dict):
        return {key: _payload(item, checkpoint=checkpoint) for key, item in value.items()}
    if isinstance(value, list):
        return [_payload(item, checkpoint=checkpoint) for item in value]
    if isinstance(value, tuple):
        return tuple(_payload(item, checkpoint=checkpoint) for item in value)
    return copy.deepcopy(value) if checkpoint else value


def _prepare(callback: Callable, argument: Any) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    with torch.no_grad():
        prepared = callback(argument)
    if not isinstance(prepared, dict):
        raise TypeError("trajectory preparation must return a payload dictionary")
    return _payload(prepared), time.perf_counter() - start


class ActiveTrajectoryPool:
    """Experimental FIFO active pool with timing-independent trajectory sampling.

    K and H are sampled once per reset from the available counts. ``advance``
    owns physical integration and carries the detached optimizer history of the
    finished payload into the next timestep; only ``reset`` clears it. ``retire``
    runs on the caller thread after backward; reset and advance run in the
    worker pool. No callback is called at restore.

    A dispatch FIFO rotates every active trajectory, including preparing ones.
    Returning a batch appends its members or their replacements to the tail.
    At 4B capacity, preparation can overlap roughly three other batches before
    its dispatch turn. A pending head may stall dispatch even if later members
    are ready; this makes fairness, batch composition, and checkpoint continuation
    independent of worker timing. No whole-pool barrier or padding is required.
    Only one batch may be checked out at a time.

    Args:
        capacity: Number of distinct active trajectories; must exceed batch_size.
        batch_size: Fixed number of trajectories in every returned batch.
        reset: Prepare an initial payload for a unique integer seed.
        advance: Prepare a new physical timestep from the solved current payload.
        retire: Release a payload's external physics context, if any.
        iteration_counts: Available positive inner iteration budgets K.
        physical_step_counts: Available positive physical timestep budgets H.
        seed: Initial unique reset seed and seed for the separate budget RNG.
        workers: Maximum concurrent preparation callbacks.
        initialize: Submit initial resets; false is used for checkpoint restore.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        reset: Callable[[int], dict[str, Any]],
        advance: Callable[[dict[str, Any]], dict[str, Any]],
        retire: Callable[[dict[str, Any]], None] | None = None,
        *,
        iteration_counts: Iterable[int],
        physical_step_counts: Iterable[int],
        seed: int,
        workers: int = 2,
        initialize: bool = True,
    ):
        self.capacity = _positive(capacity, "capacity")
        self.batch_size = _positive(batch_size, "batch_size")
        if self.capacity <= self.batch_size:
            raise ValueError("capacity must exceed batch_size")
        _positive(workers, "workers")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.set_available_counts(iteration_counts, physical_step_counts)
        self.seed = seed
        self._rng = random.Random(seed)
        self._next_seed = seed
        self._next_id = 0
        self._reset = reset
        self._advance = advance
        self._retire = retire
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trajectory-prepare")
        self._records: dict[int, TrajectoryRecord] = {}
        self._ready: deque[int] = deque()
        self._dispatch: deque[int] = deque()
        self._pending: deque[int] = deque()
        self._jobs: dict[int, tuple[Future, str]] = {}
        self._prepared: set[int] = set()
        self._batch: list[int] | None = None
        self._closed = False
        self.stats = {
            "batches": 0,
            "inner_iterations": 0,
            "advances": 0,
            "resets": 0,
            "retired": 0,
            "wait_seconds": 0.0,
            "preparation_seconds": 0.0,
        }
        if initialize:
            for _ in range(self.capacity):
                self._schedule_reset()

    @property
    def records(self) -> tuple[TrajectoryRecord, ...]:
        """Return active records; quiesce first when inspecting pending payloads."""
        return tuple(self._records.values())

    def set_available_counts(self, iteration_counts: Iterable[int], physical_step_counts: Iterable[int]) -> None:
        """Change future reset budgets without altering any active trajectory."""
        iterations = _counts(iteration_counts, "iteration_counts")
        steps = _counts(physical_step_counts, "physical_step_counts")
        self.iteration_counts = iterations
        self.physical_step_counts = steps

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("trajectory pool is closed")

    def _schedule_reset(self) -> None:
        record = TrajectoryRecord(
            id=self._next_id,
            seed=self._next_seed,
            iteration_budget=self._rng.choice(self.iteration_counts),
            step_budget=self._rng.choice(self.physical_step_counts),
        )
        self._next_id += 1
        self._next_seed += 1
        self._records[record.id] = record
        self._dispatch.append(record.id)
        self._pending.append(record.id)
        self._jobs[record.id] = (self._executor.submit(_prepare, self._reset, record.seed), "reset")
        self.stats["resets"] += 1

    def _resolve(self, identity: int) -> None:
        if identity not in self._jobs:
            return
        future, operation = self._jobs[identity]
        start = time.perf_counter()
        try:
            prepared, elapsed = future.result()
        except Exception as error:
            raise RuntimeError(f"trajectory {identity} {operation} preparation failed: {error}") from error
        finally:
            self.stats["wait_seconds"] += time.perf_counter() - start
        self._records[identity].payload = prepared
        self._prepared.add(identity)
        del self._jobs[identity]
        self.stats["preparation_seconds"] += elapsed

    def take_batch(self) -> list[TrajectoryRecord]:
        """Return exactly B distinct ready records in deterministic FIFO order."""
        self._ensure_open()
        if self._batch is not None:
            raise RuntimeError("finish the checked-out batch before taking another batch")
        if len(self._dispatch) < self.batch_size:
            raise RuntimeError("trajectory pool cannot supply a full batch")
        identities = list(islice(self._dispatch, self.batch_size))
        # Resolve the selected batch before changing queue membership. A failed
        # preparation leaves every trajectory accounted for during cleanup.
        for identity in identities:
            self._resolve(identity)
        for identity in identities:
            self._dispatch.popleft()
            if identity in self._ready:
                self._ready.remove(identity)
            else:
                self._pending.remove(identity)
        self._batch = identities
        self.stats["batches"] += 1
        return [self._records[identity] for identity in self._batch]

    def finish_batch(self, records: Iterable[TrajectoryRecord]) -> None:
        """Detach solved payloads after backward and schedule each independent boundary.

        The caller must finish all uses of the batch's autograd graph before this
        method. Final physical steps retire directly without an unused advance.
        """
        self._ensure_open()
        records = list(records)
        if (
            self._batch is None
            or [record.id for record in records] != self._batch
            or any(self._records.get(record.id) is not record for record in records)
        ):
            raise ValueError("finish_batch requires the original, complete checked-out batch in order")
        for record in records:
            record.payload = _payload(record.payload)
            record.inner_iteration += 1
            self.stats["inner_iterations"] += 1
            if record.inner_iteration < record.iteration_budget:
                self._ready.append(record.id)
                self._dispatch.append(record.id)
            elif record.physical_step + 1 < record.step_budget:
                record.inner_iteration = 0
                record.physical_step += 1
                self._pending.append(record.id)
                self._dispatch.append(record.id)
                self._jobs[record.id] = (self._executor.submit(_prepare, self._advance, record.payload), "advance")
                self.stats["advances"] += 1
            else:
                if self._retire is not None:
                    self._retire(record.payload)
                self._prepared.remove(record.id)
                del self._records[record.id]
                self.stats["retired"] += 1
                self._schedule_reset()
        self._batch = None

    def quiesce(self) -> None:
        """Wait for all submitted preparation without changing FIFO membership."""
        self._ensure_open()
        for identity in self._pending:
            self._resolve(identity)

    def state_dict(self) -> dict[str, Any]:
        """Drain preparation and snapshot detached CPU payloads with exact queue order.

        Checkpoints require a returned batch. All state is deterministic except
        ``stats.wait_seconds`` and ``stats.preparation_seconds``, which are runtime
        diagnostics. External physics contexts must be serialized separately by
        the caller; only their payload references are included here.
        """
        self._ensure_open()
        if self._batch is not None:
            raise RuntimeError("finish the checked-out batch before checkpointing")
        self.quiesce()
        return {
            "format_version": 2,
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "next_seed": self._next_seed,
            "next_id": self._next_id,
            "iteration_counts": self.iteration_counts,
            "physical_step_counts": self.physical_step_counts,
            "rng_state": self._rng.getstate(),
            "records": [_payload(vars(record), checkpoint=True) for record in self._records.values()],
            "dispatch": list(self._dispatch),
            "ready": list(self._ready),
            "pending": list(self._pending),
            "stats": dict(self.stats),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        reset: Callable[[int], dict[str, Any]],
        advance: Callable[[dict[str, Any]], dict[str, Any]],
        retire: Callable[[dict[str, Any]], None] | None = None,
        workers: int = 2,
    ) -> ActiveTrajectoryPool:
        """Restore prepared ready and pending queues without invoking callbacks."""
        if state.get("format_version") != 2:
            raise ValueError("unsupported trajectory pool checkpoint format")
        records = [TrajectoryRecord(**_payload(record, checkpoint=True)) for record in state["records"]]
        identities = [record.id for record in records]
        queues = list(state["ready"]) + list(state["pending"])
        if (
            len(records) != state["capacity"]
            or len(set(identities)) != len(identities)
            or len(queues) != len(identities)
            or set(queues) != set(identities)
            or len(state["dispatch"]) != len(identities)
            or set(state["dispatch"]) != set(identities)
            or len({record.seed for record in records}) != len(records)
        ):
            raise ValueError("invalid trajectory pool checkpoint queue membership")
        for record in records:
            _positive(record.iteration_budget, "iteration_budget")
            _positive(record.step_budget, "step_budget")
            if (
                not 0 <= record.inner_iteration < record.iteration_budget
                or not 0 <= record.physical_step < record.step_budget
                or not isinstance(record.payload, dict)
                or not 0 <= record.id < state["next_id"]
                or not state["seed"] <= record.seed < state["next_seed"]
            ):
                raise ValueError("invalid trajectory pool checkpoint record")
        rng = random.Random()
        rng.setstate(state["rng_state"])
        pool = cls(
            capacity=state["capacity"],
            batch_size=state["batch_size"],
            reset=reset,
            advance=advance,
            retire=retire,
            iteration_counts=state["iteration_counts"],
            physical_step_counts=state["physical_step_counts"],
            seed=state["seed"],
            workers=workers,
            initialize=False,
        )
        pool._rng = rng
        pool._next_id = state["next_id"]
        pool._next_seed = state["next_seed"]
        pool._records = {record.id: record for record in records}
        pool._dispatch = deque(state["dispatch"])
        pool._ready = deque(state["ready"])
        pool._pending = deque(state["pending"])
        pool._prepared = set(identities)
        pool.stats = dict(state["stats"])
        return pool

    def close(self) -> None:
        """Wait for workers and retire all known contexts, also after failures.

        The caller must finish backward before closing a checked-out batch.
        Cleanup continues after callback failures and re-raises the first error.
        Repeated calls are harmless, including after a cleanup error.
        """
        if self._closed:
            return
        first_error = None
        for identity in list(self._jobs):
            try:
                self._resolve(identity)
            except Exception as error:
                if first_error is None:
                    first_error = error
        self._executor.shutdown(wait=True)
        for identity, record in self._records.items():
            if identity in self._prepared and self._retire is not None:
                try:
                    self._retire(_payload(record.payload))
                except Exception as error:
                    if first_error is None:
                        first_error = error
        self._closed = True
        self._records.clear()
        self._prepared.clear()
        self._dispatch.clear()
        self._ready.clear()
        self._pending.clear()
        self._jobs.clear()
        self._batch = None
        if first_error is not None:
            raise first_error

    def __enter__(self) -> ActiveTrajectoryPool:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
