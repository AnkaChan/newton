# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental append-only disk retention of physical states for solver replay.

Use one SQLite file per training rank. Each committed batch is durable and
atomic; no state is evicted or overwritten. Geometry/material contexts are
content-addressed and stored once. Readers fetch only requested float32 states,
not a complete trajectory or its autograd graph. This module does not train.
"""

from __future__ import annotations

import hashlib
import io
import json
import operator
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DiskReplayDataset", "DiskReplayStore", "ReplayState"]

_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x4E495250


def _json(value: Mapping) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON mapping")
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON values") from error


def _numpy(value) -> np.ndarray:
    # Torch is optional for the disk reader. Detach before copying GPU data.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _float32(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = _numpy(value)
    if array.shape != shape or array.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating array of shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype="<f4", order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite in float32")
    return array


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass
class ReplayState:
    """A physical timestep's input, not an optimizer candidate or target label.

    Arrays may be NumPy arrays or Torch tensors on append. Reads return owned
    float32 NumPy arrays. ``forces=None`` means zero external particle force;
    ``fixed_positions=None`` means the context's rest positions at fixed indices.
    ``metadata`` holds finite JSON provenance, for example seed, physical time,
    producing checkpoint, and optimizer update. A trajectory ID must be unique
    across restart segments in its rank file; a step cannot be overwritten.
    """

    context_id: str
    trajectory_id: str
    step_index: int
    positions: Any
    velocities: Any
    forces: Any = None
    fixed_positions: Any = None
    metadata: dict = field(default_factory=dict)


class DiskReplayStore:
    """Experimental transactional storage; opening an existing file resumes it.

    A successful ``append`` commits before returning. Concurrent readers can
    observe committed batches while a rank continues appending (SQLite WAL).
    Open a separate connection in each process; do not share this object across
    DataLoader workers. Use ``close`` or a context manager to release resources.
    Use SQLite's backup API for live snapshots. Close all connections,
    including readers, before copying a standalone database file.
    """

    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path).resolve()
        self.read_only = read_only
        if read_only and not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ro" if read_only else "rwc"
        self._connection = sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True, timeout=30)
        try:
            self._initialize()
        except BaseException:
            self._connection.close()
            raise

    def _initialize(self):
        connection = self._connection
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        application = connection.execute("PRAGMA application_id").fetchone()[0]
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not tables and version == 0 and application == 0 and not self.read_only:
            with connection:
                connection.executescript(
                    f"""
                    BEGIN IMMEDIATE;
                    CREATE TABLE contexts (
                        id TEXT PRIMARY KEY,
                        metadata TEXT NOT NULL,
                        arrays BLOB NOT NULL,
                        particle_count INTEGER NOT NULL,
                        fixed_count INTEGER NOT NULL
                    );
                    CREATE TABLE states (
                        id INTEGER PRIMARY KEY,
                        context_id TEXT NOT NULL REFERENCES contexts(id),
                        trajectory_id TEXT NOT NULL,
                        step_index INTEGER NOT NULL CHECK(step_index >= 0),
                        positions BLOB NOT NULL,
                        velocities BLOB NOT NULL,
                        forces BLOB,
                        fixed_positions BLOB,
                        metadata TEXT NOT NULL,
                        UNIQUE (trajectory_id, step_index)
                    );
                    CREATE INDEX states_context ON states(context_id, id);
                    PRAGMA user_version={_SCHEMA_VERSION};
                    PRAGMA application_id={_APPLICATION_ID};
                    COMMIT;
                    """
                )
        elif version != _SCHEMA_VERSION or application != _APPLICATION_ID:
            raise ValueError("unrecognized replay database or unsupported schema version")
        connection.execute("PRAGMA foreign_keys=ON")
        if not self.read_only:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Release the connection and allow SQLite to checkpoint its WAL."""
        self._connection.close()

    def _writable(self):
        if self.read_only:
            raise PermissionError("replay store is read-only")

    def register_context(self, *, metadata: Mapping, arrays: Mapping[str, Any]) -> str:
        """Deduplicate immutable geometry/material arrays and finite JSON metadata.

        Required arrays: ``rest_positions[P,3]`` and ``fixed_indices[F]``.
        Additional numerical arrays can store topology and per-cell materials.
        The context ID hashes canonical metadata, array names, dtypes, shapes,
        and bytes. No Python pickle is stored or loaded.
        """
        self._writable()
        encoded_metadata = _json(metadata)
        if not isinstance(arrays, Mapping) or not all(isinstance(name, str) and name for name in arrays):
            raise ValueError("arrays must map nonempty names to numerical arrays")
        normalized = {}
        digest = hashlib.sha256(encoded_metadata.encode())
        for name, value in sorted(arrays.items()):
            array = _numpy(value)
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError(f"context array {name} must be finite and numerical")
            array = np.array(array, dtype=array.dtype.newbyteorder("<"), copy=True, order="C")
            description = json.dumps([name, array.dtype.str, array.shape], separators=(",", ":")).encode()
            digest.update(len(description).to_bytes(8, "little"))
            digest.update(description)
            digest.update(array.tobytes())
            normalized[name] = array
        rest = normalized.get("rest_positions")
        fixed = normalized.get("fixed_indices")
        if rest is None or rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 1 or rest.dtype.kind != "f":
            raise ValueError("context rest_positions must be floating [P,3]")
        if (
            fixed is None
            or fixed.ndim != 1
            or fixed.dtype.kind not in "iu"
            or np.any(fixed < 0)
            or np.any(fixed >= len(rest))
            or len(np.unique(fixed)) != len(fixed)
        ):
            raise ValueError("context fixed_indices must be unique indices within rest_positions")
        stream = io.BytesIO()
        np.savez_compressed(stream, **normalized)
        context_id = digest.hexdigest()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO contexts VALUES (?, ?, ?, ?, ?)",
                (context_id, encoded_metadata, stream.getvalue(), len(rest), len(fixed)),
            )
        return context_id

    def get_context(self, context_id: str) -> dict:
        """Load one context; returned arrays and metadata are owned by the caller."""
        row = self._connection.execute("SELECT metadata, arrays FROM contexts WHERE id=?", (context_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown replay context: {context_id}")
        with np.load(io.BytesIO(row[1]), allow_pickle=False) as arrays:
            return {
                "context_id": context_id,
                "metadata": json.loads(row[0]),
                "arrays": {name: arrays[name].copy() for name in arrays.files},
            }

    def append(self, states: Sequence[ReplayState]) -> tuple[int, ...]:
        """Commit a batch of timestep starts, atomically and without eviction.

        Duplicate (trajectory_id, step_index) raises ``sqlite3.IntegrityError``
        and rolls back the whole batch. Unknown contexts or invalid/nonfinite
        inputs also leave the store unchanged. Float64 inputs are converted to
        float32; already-float32 values round-trip exactly.
        """
        self._writable()
        rows = []
        dimensions = {}
        trajectories = {}
        for state in states:
            if not isinstance(state, ReplayState):
                raise TypeError("states must contain ReplayState records")
            if not isinstance(state.trajectory_id, str) or not state.trajectory_id:
                raise ValueError("trajectory_id must be nonempty")
            step = _nonnegative_int(state.step_index, "step_index")
            if state.trajectory_id not in trajectories:
                previous = self._connection.execute(
                    "SELECT context_id FROM states WHERE trajectory_id=? LIMIT 1", (state.trajectory_id,)
                ).fetchone()
                trajectories[state.trajectory_id] = state.context_id if previous is None else previous[0]
            if trajectories[state.trajectory_id] != state.context_id:
                raise ValueError("a trajectory must retain its original physical context")
            if state.context_id not in dimensions:
                found = self._connection.execute(
                    "SELECT particle_count, fixed_count FROM contexts WHERE id=?", (state.context_id,)
                ).fetchone()
                if found is None:
                    raise KeyError(f"unknown replay context: {state.context_id}")
                dimensions[state.context_id] = found
            particles, fixed = dimensions[state.context_id]
            shape = (particles, 3)
            positions = _float32(state.positions, shape, "positions").tobytes()
            velocities = _float32(state.velocities, shape, "velocities").tobytes()
            forces = None
            if state.forces is not None:
                values = _float32(state.forces, shape, "forces")
                if np.any(values):
                    forces = values.tobytes()
            pins = (
                None
                if state.fixed_positions is None
                else _float32(state.fixed_positions, (fixed, 3), "fixed_positions").tobytes()
            )
            rows.append(
                (
                    state.context_id,
                    state.trajectory_id,
                    step,
                    positions,
                    velocities,
                    forces,
                    pins,
                    _json(state.metadata),
                )
            )
        identifiers = []
        with self._connection:
            for row in rows:
                cursor = self._connection.execute(
                    "INSERT INTO states (context_id,trajectory_id,step_index,positions,velocities,forces,"
                    "fixed_positions,metadata) VALUES (?,?,?,?,?,?,?,?)",
                    row,
                )
                identifiers.append(cursor.lastrowid)
        return tuple(identifiers)

    def get(self, record_id: int) -> ReplayState:
        """Fetch one committed record without loading other states."""
        record_id = _nonnegative_int(record_id, "record_id")
        row = self._connection.execute(
            "SELECT s.context_id,s.trajectory_id,s.step_index,s.positions,s.velocities,s.forces,s.fixed_positions,"
            "s.metadata,c.particle_count,c.fixed_count FROM states s JOIN contexts c ON s.context_id=c.id WHERE s.id=?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown replay record: {record_id}")

        def array(data, count):
            return None if data is None else np.frombuffer(data, dtype="<f4").reshape(count, 3).copy()

        return ReplayState(
            context_id=row[0],
            trajectory_id=row[1],
            step_index=row[2],
            positions=array(row[3], row[8]),
            velocities=array(row[4], row[8]),
            forces=array(row[5], row[8]),
            fixed_positions=array(row[6], row[9]),
            metadata=json.loads(row[7]),
        )

    def record_ids(self, *, context_id: str | None = None) -> np.ndarray:
        """Return a compact ordered index of committed rows, without state blobs."""
        if context_id is None:
            rows = self._connection.execute("SELECT id FROM states ORDER BY id")
        else:
            rows = self._connection.execute("SELECT id FROM states WHERE context_id=? ORDER BY id", (context_id,))
        return np.fromiter((row[0] for row in rows), dtype=np.int64)

    def __len__(self):
        return self._connection.execute("SELECT count(*) FROM states").fetchone()[0]


class DiskReplayDataset:
    """Experimental lazy replay across rank files with an explicit index snapshot.

    Sampling holds its population fixed until ``refresh``; newly appended data
    cannot silently change a seeded sample. The index uses eight bytes per
    retained state; state arrays stay on disk until selected. Use one dataset
    object per process. Paths are sorted so discovery order is immaterial.
    ``context_id`` can restrict sampling to one compatible geometry/material.
    """

    def __init__(self, paths: Sequence[str | Path], *, context_id: str | None = None):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = tuple(sorted({Path(path).resolve() for path in paths}))
        if not self.paths:
            raise ValueError("at least one replay path is required")
        self.context_id = context_id
        self._stores = []
        try:
            for path in self.paths:
                self._stores.append(DiskReplayStore(path, read_only=True))
            self.refresh()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Close all rank readers."""
        for store in self._stores:
            store.close()

    def refresh(self):
        """Explicitly include currently committed states in the sample population."""
        self._indices = [store.record_ids(context_id=self.context_id) for store in self._stores]
        self._ends = np.cumsum([len(index) for index in self._indices], dtype=np.int64)

    def __len__(self):
        return int(self._ends[-1])

    def __getitem__(self, index: int) -> ReplayState:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rank = int(np.searchsorted(self._ends, index, side="right"))
        offset = index - (int(self._ends[rank - 1]) if rank else 0)
        return self._stores[rank].get(int(self._indices[rank][offset]))

    def get_context(self, context_id: str) -> dict:
        """Read a shared context from the first rank that contains it."""
        for store in self._stores:
            try:
                return store.get_context(context_id)
            except KeyError:
                pass
        raise KeyError(f"unknown replay context: {context_id}")

    def sample(self, count: int, *, seed: int, replace: bool = False) -> list[ReplayState]:
        """Uniform seeded replay; no implicit eviction, recency weighting, or GPU copy."""
        count = _nonnegative_int(count, "count")
        seed = _nonnegative_int(seed, "seed")
        if not isinstance(replace, bool):
            raise ValueError("replace must be boolean")
        indices = np.random.default_rng(seed).choice(len(self), size=count, replace=replace)
        return [self[int(index)] for index in indices]
