# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Deterministic durable artifacts for trusted PR-history v5 samples.

The source JSON is metadata and evidence, never executable authority.  The
reader first verifies externally pinned file hashes, validates a bounded NPZ
container, reconstructs source-bound PR dataclasses and physics from the exact
numeric arrays, and reruns :func:`load_source_bound_pr_history_v5_sample`.
Persisted sample and acceptance records are accepted only when that independent
reconstruction reproduces them.  A fresh current-code history is compared only
as a compatibility diagnostic; ordinary checkout changes cannot invalidate an
otherwise byte-valid, externally anchored archive.

No split role is selected here.  The returned :class:`TrajectoryRecord` can be
placed in a train/validation/confirmation split only by a separate frozen
split manifest and access ledger.  This bounded format persists and exposes a
complete accepted chain only; subrange archives require a future prefix-proof,
selection-only format.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import pathlib
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import torch

from .correction_ceiling import _reconstruct_canonical_history, _snapshot_chain
from .pr_scene_history import (
    AppliedAtomicState,
    AtomicCoordinate,
    CommittedState,
    HistoryCheckpoint,
    HistoryTransition,
    PRHistoryChain,
    PRHistoryManifest,
    PRHistoryStaticBundle,
    PRSceneHistory,
    TransitionTiming,
    _array_digest,
)
from .solver_benchmark import TetBenchmarkScene
from .train_pr_history_v5 import canonical_training_tensor_sha256
from .v5_checkpoint import canonical_json_sha256
from .v5_dataset import TrajectoryProvenance, TrajectoryRecord
from .v5_pr_history_loader import (
    LoadedPRHistoryV5Sample,
    load_pr_history_v5_sample,
    load_source_bound_pr_history_v5_sample,
)

_SOURCE_CONTRACT = "pss-v5-pr-history-artifact-source-v3"
_BUNDLE_CONTRACT = "pss-v5-pr-history-deterministic-npz-v1"
_GENERATION_CONTRACT = "pss-v5-pr-history-generation-spec-v1"
_PIN_SCHEDULE_CONTRACT = "pss-v5-pr-history-pin-schedule-v1"
_EVENT_INVENTORY_CONTRACT = "pss-v5-pr-history-event-inventory-v1"
_COORDINATE_RANGE_CONTRACT = "pss-v5-pr-history-coordinate-range-v1"
_SELECTION_CONTRACT = "pss-v5-pr-history-contiguous-selection-v2"
_LOAD_PROGRAM_CONTRACT = "pss-v5-pr-history-load-program-v1"
_NPY_VERSION = (2, 0)
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_STATIC_ARRAY_NAMES = (
    "rest_q",
    "tet_indices",
    "tet_poses",
    "mass",
    "tet_materials",
    "gravity",
    "external_force",
)
_BASE_SCENE_ARRAY_NAMES = (
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
    "x_current",
    "velocity",
    "gravity",
    "external_force",
    "pinned_indices",
    "pin_targets",
)
_BASE_SCENE_DTYPES = {
    name: np.dtype(
        np.int32
        if name == "particle_flags"
        else np.int64
        if name
        in {
            "tet_indices",
            "tri_indices",
            "color_group_offsets",
            "color_group_particles",
            "pinned_indices",
        }
        else np.float64
    )
    for name in _BASE_SCENE_ARRAY_NAMES
}
_STATIC_DTYPES = {
    "rest_q": np.dtype(np.float32),
    "tet_indices": np.dtype(np.int64),
    "tet_poses": np.dtype(np.float32),
    "mass": np.dtype(np.float32),
    "tet_materials": np.dtype(np.float32),
    "gravity": np.dtype(np.float32),
    "external_force": np.dtype(np.float32),
}
_STATE_DTYPES = {
    "q": np.dtype(np.float32),
    "qd": np.dtype(np.float32),
    "particle_flags": np.dtype(np.int32),
}
_TRANSITION_DTYPES = {
    "applied_q": np.dtype(np.float32),
    "applied_particle_flags": np.dtype(np.int32),
    "pinned_indices": np.dtype(np.int64),
    "pin_targets": np.dtype(np.float32),
    "inertial_target": np.dtype(np.float32),
    "reference_positions": np.dtype(np.float64),
}
_BASE_SCENE_RANKS = {
    name: (
        3
        if name in {"tet_poses", "tri_poses"}
        else 2
        if name
        in {
            "rest_q",
            "tet_indices",
            "tet_materials",
            "tri_indices",
            "tri_materials",
            "x_current",
            "velocity",
            "external_force",
            "pin_targets",
        }
        else 1
    )
    for name in _BASE_SCENE_ARRAY_NAMES
}
_STATIC_RANKS = {
    "rest_q": 2,
    "tet_indices": 2,
    "tet_poses": 3,
    "mass": 1,
    "tet_materials": 2,
    "gravity": 1,
    "external_force": 2,
}
_STATE_RANKS = {"q": 2, "qd": 2, "particle_flags": 1}
_TRANSITION_RANKS = {
    "applied_q": 2,
    "applied_particle_flags": 1,
    "pinned_indices": 1,
    "pin_targets": 2,
    "inertial_target": 2,
    "reference_positions": 2,
}
_MAX_NPY_HEADER_BYTES = 64 * 1024
_SAMPLE_WRAPPER_KEYS = {
    "sample_record",
    "reference_acceptance",
    "physical_integration",
    "operator_geometry",
    "loaded_sample_sha256",
}
_SOURCE_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract",
    "bundle_contract",
    "npy_version",
    "zip",
    "trajectory_id",
    "artifact_uris",
    "bundle_sha256",
    "history_kind",
    "history_manifest",
    "static_bundle",
    "base_scene",
    "chain",
    "arrays",
    "selection",
    "provenance_ingredients",
    "samples",
    "trust_scope",
    "source_record_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int/float equality aliases."""
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _file_sha256(path: pathlib.Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("durable PR-history arrays must not use object dtype")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError("durable PR-history arrays must be finite")
    dtype = array.dtype if array.dtype.byteorder == "|" else array.dtype.newbyteorder("<")
    return np.array(array, dtype=dtype, order="C", copy=True)


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = _canonical_array(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_digest(array),
        "nbytes": array.nbytes,
    }


def _require_array_dtype(value: np.ndarray, dtype: np.dtype | type, name: str) -> None:
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"durable array {name} must have dtype {np.dtype(dtype)}")


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, _canonical_array(value), version=_NPY_VERSION, allow_pickle=False)
    return stream.getvalue()


def _validate_destination(path: str | pathlib.Path, name: str) -> pathlib.Path:
    destination = pathlib.Path(path).resolve()
    if not destination.parent.is_dir():
        raise ValueError(f"{name} parent directory must already exist")
    if os.path.lexists(destination):
        raise FileExistsError(f"{name} already exists: {destination}")
    return destination


def _validate_uri(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{name} must be a non-empty canonical URI")
    parsed = urlsplit(value)
    if not parsed.scheme or (not parsed.netloc and not parsed.path) or not value.startswith(f"{parsed.scheme}:"):
        raise ValueError(f"{name} must be an absolute URI with canonical lowercase scheme spelling")
    return value


def _temporary_path(destination: pathlib.Path) -> pathlib.Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    return pathlib.Path(raw_path)


def _publish_without_overwrite(temporary: pathlib.Path, destination: pathlib.Path) -> None:
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_canonical_zip(stream, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            if (
                not name
                or name.startswith("/")
                or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
            ):
                raise ValueError(f"noncanonical durable array name {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))


def _write_bundle(destination: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> str:
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w+b") as stream:
            _write_canonical_zip(stream, arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _file_sha256(temporary)
        _publish_without_overwrite(temporary, destination)
        return digest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_source(destination: pathlib.Path, source: Mapping[str, object]) -> str:
    temporary = _temporary_path(destination)
    try:
        payload = _canonical_json_bytes(source) + b"\n"
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(payload).hexdigest()
        _publish_without_overwrite(temporary, destination)
        return digest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _state_array_prefix(ordinal: int) -> str:
    return f"state/{ordinal:06d}"


def _transition_array_prefix(ordinal: int) -> str:
    return f"transition/{ordinal:06d}"


def _expected_array_specs(transition_count: int) -> dict[str, tuple[np.dtype, int]]:
    if type(transition_count) is not int or transition_count < 1:
        raise ValueError("durable artifact requires at least one accepted transition")
    expected = {f"base_scene/{name}": (dtype, _BASE_SCENE_RANKS[name]) for name, dtype in _BASE_SCENE_DTYPES.items()}
    expected.update({f"static/{name}": (dtype, _STATIC_RANKS[name]) for name, dtype in _STATIC_DTYPES.items()})
    for ordinal in range(transition_count + 1):
        prefix = _state_array_prefix(ordinal)
        expected.update({f"{prefix}/{name}": (dtype, _STATE_RANKS[name]) for name, dtype in _STATE_DTYPES.items()})
    for ordinal in range(transition_count):
        prefix = _transition_array_prefix(ordinal)
        expected.update(
            {f"{prefix}/{name}": (dtype, _TRANSITION_RANKS[name]) for name, dtype in _TRANSITION_DTYPES.items()}
        )
    return expected


def _collect_arrays(history: PRSceneHistory, chain: PRHistoryChain) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in _BASE_SCENE_ARRAY_NAMES:
        arrays[f"base_scene/{name}"] = _canonical_array(getattr(history._base_scene, name))
    for name in _STATIC_ARRAY_NAMES:
        arrays[f"static/{name}"] = _canonical_array(getattr(history.static_bundle, name))
    states = (chain.initial_checkpoint.state, *(transition.output_state for transition in chain.transitions))
    for state_value in states:
        prefix = _state_array_prefix(state_value.coordinate.ordinal)
        arrays[f"{prefix}/q"] = _canonical_array(state_value.q)
        arrays[f"{prefix}/qd"] = _canonical_array(state_value.qd)
        arrays[f"{prefix}/particle_flags"] = _canonical_array(state_value.particle_flags)
    for transition in chain.transitions:
        prefix = _transition_array_prefix(transition.coordinate.ordinal)
        applied = transition.applied_state
        arrays[f"{prefix}/applied_q"] = _canonical_array(applied.q)
        arrays[f"{prefix}/applied_particle_flags"] = _canonical_array(applied.particle_flags)
        arrays[f"{prefix}/pinned_indices"] = _canonical_array(applied.pinned_indices)
        arrays[f"{prefix}/pin_targets"] = _canonical_array(applied.pin_targets)
        arrays[f"{prefix}/inertial_target"] = _canonical_array(transition.inertial_target)
        arrays[f"{prefix}/reference_positions"] = _canonical_array(transition.reference_positions)
    if len(arrays) != len(set(arrays)):
        raise RuntimeError("internal durable array names are not unique")
    if set(arrays) != set(_expected_array_specs(len(chain.transitions))):
        raise RuntimeError("internal durable array inventory differs from the canonical chain-derived inventory")
    return arrays


def _chain_record_without_timings(chain: PRHistoryChain) -> dict[str, object]:
    record = chain.as_dict()
    record.pop("timings")
    record["accepted_states"] = [
        chain.initial_checkpoint.state.as_dict(),
        *(transition.output_state.as_dict() for transition in chain.transitions),
    ]
    record["timing_policy"] = "excluded-nondeterministic-diagnostics-v1"
    return record


def _coordinate_record(coordinate: AtomicCoordinate) -> dict[str, int]:
    return coordinate.as_dict()


def _provenance_ingredients(
    history_kind: str,
    manifest: PRHistoryManifest,
    base_scene: TetBenchmarkScene,
    chain: PRHistoryChain,
) -> dict[str, object]:
    initial_velocity = chain.initial_checkpoint.state.qd
    if not np.array_equal(initial_velocity, np.broadcast_to(initial_velocity[0], initial_velocity.shape)):
        raise ValueError("PR root velocity is not one uniform provenance vector")
    generation_spec = {
        "contract": _GENERATION_CONTRACT,
        "generator": "PRSceneHistory.generate",
        "history_kind": history_kind,
        "manifest_sha256": manifest.manifest_sha256,
        "root_checkpoint_sha256": chain.initial_checkpoint.checkpoint_sha256,
        "final_checkpoint_sha256": chain.final_checkpoint.checkpoint_sha256,
        "accepted_transition_count": len(chain.transitions),
        "start": _coordinate_record(chain.initial_checkpoint.state.coordinate),
        "stop": _coordinate_record(chain.final_checkpoint.state.coordinate),
        "reference_methods": [str(transition.reference_record["method"]) for transition in chain.transitions],
        "reference_configs": [dict(transition.reference_record["config"]) for transition in chain.transitions],
        "generation_seed": 0,
        "randomness": "none",
        "timings": "excluded",
    }
    pin_schedule = {
        "contract": _PIN_SCHEDULE_CONTRACT,
        "transitions": [
            {
                "ordinal": transition.coordinate.ordinal,
                "applied_sha256": transition.applied_state.applied_sha256,
                "pinned_indices_sha256": _array_digest(transition.applied_state.pinned_indices),
                "pin_targets_sha256": _array_digest(transition.applied_state.pin_targets),
            }
            for transition in chain.transitions
        ],
    }
    event_inventory = {
        "contract": _EVENT_INVENTORY_CONTRACT,
        "events": [
            {
                "ordinal": transition.coordinate.ordinal,
                "callback_applied": transition.applied_state.callback_applied,
                "action": transition.applied_state.action,
                "schedule_value_name": transition.applied_state.schedule_value_name,
                "schedule_value": transition.applied_state.schedule_value,
            }
            for transition in chain.transitions
        ],
    }
    start = _coordinate_record(chain.initial_checkpoint.state.coordinate)
    stop = _coordinate_record(chain.final_checkpoint.state.coordinate)
    coordinate_range = {
        "contract": _COORDINATE_RANGE_CONTRACT,
        "start": start,
        "stop": stop,
        "accepted_transition_count": len(chain.transitions),
        "semantics": "start-inclusive-stop-exclusive",
    }
    density = base_scene.metadata.get("density_kg_m3")
    if isinstance(density, bool) or not isinstance(density, (int, float)) or not np.isfinite(density):
        raise ValueError("canonical PR history is missing finite density provenance")
    return {
        "generation_spec": generation_spec,
        "generation_spec_sha256": canonical_json_sha256(generation_spec),
        "pin_schedule_sha256": canonical_json_sha256(pin_schedule),
        "event_inventory_sha256": canonical_json_sha256(event_inventory),
        "coordinate_start_sha256": canonical_json_sha256(start),
        "coordinate_stop_sha256": canonical_json_sha256(stop),
        "coordinate_range_sha256": canonical_json_sha256(coordinate_range),
        "density_kg_m3": float(density),
        "initial_velocity_m_s": [float(value) for value in initial_velocity[0]],
        "generation_seed": 0,
    }


def _selection_record(
    chain: PRHistoryChain,
    loaded_samples: Sequence[LoadedPRHistoryV5Sample],
    start: int,
    stop: int,
) -> dict[str, object]:
    selected_transitions = chain.transitions[start:stop]
    payload = {
        "contract": _SELECTION_CONTRACT,
        "source_chain_sha256": chain.chain_sha256,
        "source_transition_count": len(chain.transitions),
        "selected_start_ordinal": start,
        "selected_stop_ordinal_exclusive": stop,
        "excluded_prefix_count": start,
        "excluded_suffix_count": len(chain.transitions) - stop,
        "selected_transition_sha256": [transition.transition_sha256 for transition in selected_transitions],
        "selected_sample_sha256": [loaded.training_sample.sample_record.sample_sha256 for loaded in loaded_samples],
        "selected_reference_acceptance_sha256": [
            loaded.reference_acceptance.acceptance_sha256 for loaded in loaded_samples
        ],
        "selected_physical_integration_binding_sha256": [
            loaded.physical_integration.binding_sha256 for loaded in loaded_samples
        ],
        "selected_operator_geometry_sha256": [
            loaded.training_sample.sample_record.operator_geometry_sha256 for loaded in loaded_samples
        ],
        "selected_source_tet_poses_sha256": [
            _operator_geometry_record(loaded)["source_tet_poses_sha256"] for loaded in loaded_samples
        ],
    }
    return {**payload, "selection_sha256": canonical_json_sha256(payload)}


def _operator_geometry_record(loaded: LoadedPRHistoryV5Sample) -> dict[str, object]:
    sample = loaded.training_sample
    state = sample.projection_state
    source_tet_poses = state.source_tet_poses
    source_tet_poses_array = source_tet_poses.detach().cpu().numpy()
    record = {
        "sample_id": sample.sample_record.sample_id,
        "operator_geometry_policy": state.operator_geometry_policy,
        "operator_geometry_sha256": state.operator_geometry_sha256,
        "sample_operator_geometry_sha256": sample.sample_record.operator_geometry_sha256,
        "source_tet_poses_sha256": canonical_training_tensor_sha256(source_tet_poses),
        "source_tet_poses_array_sha256": _array_digest(source_tet_poses_array),
        "source_tet_poses_dtype": str(source_tet_poses.dtype),
        "source_tet_poses_shape": list(source_tet_poses.shape),
    }
    if record["operator_geometry_sha256"] != record["sample_operator_geometry_sha256"]:
        raise ValueError("loaded sample record does not bind its projection operator geometry")
    return record


def _base_scene_record(scene: TetBenchmarkScene) -> dict[str, object]:
    manifest = scene.manifest()
    return {
        "name": scene.name,
        "source": scene.source,
        "dt_seconds": scene.dt,
        "metadata": manifest["metadata"],
        "manifest": manifest,
    }


def _trust_scope_record() -> dict[str, bool]:
    return {
        "persisted_acceptance_is_authority": False,
        "reader_must_reconstruct_chain": True,
        "reader_must_reconstruct_source_bound_physics": True,
        "reader_must_rerun_source_bound_trusted_sample_loader": True,
        "source_integration_replayed_from_exact_archival_arrays": True,
        "current_code_reproduction_is_load_prerequisite": False,
        "current_code_compatibility_reported_separately": True,
        "complete_chain_selection_required": True,
        "excluded_transition_payload_persisted": False,
        "timing_diagnostics_persisted": False,
        "split_role_assigned": False,
    }


def _source_payload(
    kind: str,
    manifest: PRHistoryManifest,
    static_bundle: PRHistoryStaticBundle,
    base_scene: TetBenchmarkScene,
    chain: PRHistoryChain,
    loaded_samples: Sequence[LoadedPRHistoryV5Sample],
    arrays: Mapping[str, np.ndarray],
    selection: Mapping[str, object],
    provenance_ingredients: Mapping[str, object],
    *,
    trajectory_id: str,
    bundle_uri: str,
    source_uri: str,
    bundle_sha256: str,
) -> dict[str, object]:
    payload = {
        "schema_version": 3,
        "contract": _SOURCE_CONTRACT,
        "bundle_contract": _BUNDLE_CONTRACT,
        "npy_version": list(_NPY_VERSION),
        "zip": {
            "compression": "stored",
            "timestamp": list(_ZIP_TIMESTAMP),
            "entry_order": "lexicographic",
        },
        "trajectory_id": trajectory_id,
        "artifact_uris": {"bundle": bundle_uri, "source": source_uri},
        "bundle_sha256": bundle_sha256,
        "history_kind": kind,
        "history_manifest": manifest.as_dict(),
        "static_bundle": static_bundle.as_dict(),
        "base_scene": _base_scene_record(base_scene),
        "chain": _chain_record_without_timings(chain),
        "arrays": {name: _array_record(value) for name, value in sorted(arrays.items())},
        "selection": dict(selection),
        "provenance_ingredients": dict(provenance_ingredients),
        "samples": [
            {
                "sample_record": loaded.training_sample.sample_record.as_dict(),
                "reference_acceptance": loaded.reference_acceptance.as_dict(),
                "physical_integration": loaded.physical_integration.as_dict(),
                "operator_geometry": _operator_geometry_record(loaded),
                "loaded_sample_sha256": loaded.loaded_sample_sha256,
            }
            for loaded in loaded_samples
        ],
        "trust_scope": _trust_scope_record(),
    }
    return {**payload, "source_record_sha256": canonical_json_sha256(payload)}


def _build_dataset_record(
    source: Mapping[str, object],
    loaded_samples: Sequence[LoadedPRHistoryV5Sample],
    *,
    source_file_sha256: str,
    bundle_file_sha256: str,
) -> TrajectoryRecord:
    ingredients = source["provenance_ingredients"]
    if not isinstance(ingredients, Mapping):
        raise ValueError("source provenance ingredients must be a mapping")
    chain = source["chain"]
    static = source["static_bundle"]
    uris = source["artifact_uris"]
    selection = source["selection"]
    if not all(isinstance(value, Mapping) for value in (chain, static, uris, selection)):
        raise ValueError("source dataset records have invalid container types")
    provenance = TrajectoryProvenance(
        generation_spec_sha256=str(ingredients["generation_spec_sha256"]),
        history_manifest_sha256=str(source["history_manifest"]["manifest_sha256"]),
        root_checkpoint_sha256=str(chain["initial_checkpoint"]["checkpoint_sha256"]),
        final_checkpoint_sha256=str(chain["final_checkpoint"]["checkpoint_sha256"]),
        artifact_bundle_uri=str(uris["bundle"]),
        artifact_bundle_sha256=bundle_file_sha256,
        artifact_source_uri=str(uris["source"]),
        artifact_source_sha256=source_file_sha256,
        static_bundle_sha256=str(static["static_sha256"]),
        density_kg_m3=float(ingredients["density_kg_m3"]),
        initial_velocity_m_s=tuple(ingredients["initial_velocity_m_s"]),
        pin_schedule_sha256=str(ingredients["pin_schedule_sha256"]),
        event_inventory_sha256=str(ingredients["event_inventory_sha256"]),
        coordinate_start_sha256=str(ingredients["coordinate_start_sha256"]),
        coordinate_stop_sha256=str(ingredients["coordinate_stop_sha256"]),
        coordinate_range_sha256=str(ingredients["coordinate_range_sha256"]),
        dt_seconds=float(source["history_manifest"]["dt_seconds"]),
        generation_seed=int(ingredients["generation_seed"]),
    )
    sample_records = tuple(loaded.training_sample.sample_record for loaded in loaded_samples)
    first = sample_records[0]
    if any(sample.topology_sha256 != first.topology_sha256 for sample in sample_records):
        raise ValueError("selected durable samples do not share one topology")
    if any(sample.material_sha256 != first.material_sha256 for sample in sample_records):
        raise ValueError("selected durable samples do not share one material identity")
    if any(sample.operator_geometry_sha256 != first.operator_geometry_sha256 for sample in sample_records):
        raise ValueError("selected durable samples do not share one operator geometry")
    source_transition_count = int(selection["source_transition_count"])
    complete = (
        int(selection["selected_start_ordinal"]) == 0
        and int(selection["selected_stop_ordinal_exclusive"]) == source_transition_count
    )
    if not complete:
        raise ValueError("durable PR-history artifacts require complete-chain selection")
    load_program = {
        "contract": _LOAD_PROGRAM_CONTRACT,
        "history_kind": source["history_kind"],
        "history_manifest_sha256": source["history_manifest"]["manifest_sha256"],
        "source_chain_sha256": chain["chain_sha256"],
    }
    return TrajectoryRecord(
        trajectory_id=str(source["trajectory_id"]),
        scene_family=f"pr2901-{source['history_kind']}",
        load_program_id=f"pr2901:{source['history_kind']}:{str(chain['chain_sha256'])[:16]}",
        load_program_sha256=canonical_json_sha256(load_program),
        source_chain_sha256=str(chain["chain_sha256"]),
        topology_sha256=first.topology_sha256,
        operator_geometry_sha256=first.operator_geometry_sha256,
        material_sha256=first.material_sha256,
        provenance=provenance,
        source_transition_count=source_transition_count,
        samples=sample_records,
        selection_contract="complete-contiguous-trajectory-v1",
        selection_provenance_sha256=None,
    )


@dataclasses.dataclass(frozen=True)
class BuiltPRV5Artifact:
    """New deterministic files and their reconstructed dataset record."""

    bundle_path: pathlib.Path
    source_path: pathlib.Path
    bundle_file_sha256: str
    source_file_sha256: str
    source_record_sha256: str
    loaded_samples: tuple[LoadedPRHistoryV5Sample, ...]
    trajectory: TrajectoryRecord


@dataclasses.dataclass(frozen=True)
class SourceBoundPRHistory:
    """Archive-reconstructed PR identity independent of the current checkout."""

    kind: str
    manifest: PRHistoryManifest
    static_bundle: PRHistoryStaticBundle
    initial_checkpoint: HistoryCheckpoint
    base_scene: TetBenchmarkScene


@dataclasses.dataclass(frozen=True)
class CurrentCodeCompatibility:
    """Non-authoritative comparison with a fresh history from current code."""

    compatible: bool
    manifest_matches: bool
    static_bundle_matches: bool
    root_checkpoint_matches: bool
    base_scene_matches: bool
    current_manifest_sha256: str | None
    current_root_checkpoint_sha256: str | None
    reason: str | None


@dataclasses.dataclass(frozen=True)
class LoadedPRV5Artifact:
    """Sealed archival evidence without exposing mutable raw PR arrays."""

    bundle_path: pathlib.Path
    source_path: pathlib.Path
    bundle_file_sha256: str
    source_file_sha256: str
    source_record_sha256: str
    history_manifest_sha256: str
    root_checkpoint_sha256: str
    source_chain_sha256: str
    static_bundle_sha256: str
    base_scene_sha256: str
    loaded_samples: tuple[LoadedPRHistoryV5Sample, ...]
    trajectory: TrajectoryRecord
    current_code_compatibility: CurrentCodeCompatibility
    artifact_evidence_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_path", self.bundle_path.resolve())
        object.__setattr__(self, "source_path", self.source_path.resolve())
        object.__setattr__(self, "loaded_samples", tuple(self.loaded_samples))
        self._validate_contents()
        object.__setattr__(self, "artifact_evidence_sha256", canonical_json_sha256(self._payload()))

    def _validate_contents(self) -> None:
        for name in (
            "bundle_file_sha256",
            "source_file_sha256",
            "source_record_sha256",
            "history_manifest_sha256",
            "root_checkpoint_sha256",
            "source_chain_sha256",
            "static_bundle_sha256",
            "base_scene_sha256",
        ):
            _sha256(getattr(self, name), name)
        if not self.loaded_samples or any(type(item) is not LoadedPRHistoryV5Sample for item in self.loaded_samples):
            raise ValueError("loaded artifact requires canonical trainer samples")
        for loaded in self.loaded_samples:
            loaded.validate_immutable()
        if type(self.trajectory) is not TrajectoryRecord:
            raise ValueError("loaded artifact trajectory must be a canonical TrajectoryRecord")
        if self.trajectory.trajectory_sha256 != canonical_json_sha256(self.trajectory._payload()):
            raise ValueError("loaded artifact trajectory changed after authentication")
        sample_records = tuple(item.training_sample.sample_record.as_dict() for item in self.loaded_samples)
        if sample_records != tuple(item.as_dict() for item in self.trajectory.samples):
            raise ValueError("loaded artifact samples differ from the authenticated trajectory")
        provenance = self.trajectory.provenance
        if (
            self.trajectory.source_chain_sha256 != self.source_chain_sha256
            or provenance.history_manifest_sha256 != self.history_manifest_sha256
            or provenance.root_checkpoint_sha256 != self.root_checkpoint_sha256
            or provenance.static_bundle_sha256 != self.static_bundle_sha256
            or provenance.artifact_bundle_sha256 != self.bundle_file_sha256
            or provenance.artifact_source_sha256 != self.source_file_sha256
        ):
            raise ValueError("loaded artifact source identities differ from trajectory provenance")
        if type(self.current_code_compatibility) is not CurrentCodeCompatibility:
            raise ValueError("loaded artifact compatibility result has a noncanonical type")

    def _payload(self) -> dict[str, object]:
        return {
            "contract": "pss-v5-loaded-pr-history-artifact-evidence-v1",
            "bundle_path": str(self.bundle_path),
            "source_path": str(self.source_path),
            "bundle_file_sha256": self.bundle_file_sha256,
            "source_file_sha256": self.source_file_sha256,
            "source_record_sha256": self.source_record_sha256,
            "history_manifest_sha256": self.history_manifest_sha256,
            "root_checkpoint_sha256": self.root_checkpoint_sha256,
            "source_chain_sha256": self.source_chain_sha256,
            "static_bundle_sha256": self.static_bundle_sha256,
            "base_scene_sha256": self.base_scene_sha256,
            "loaded_sample_sha256": [item.loaded_sample_sha256 for item in self.loaded_samples],
            "trajectory_sha256": self.trajectory.trajectory_sha256,
            "current_code_compatibility": dataclasses.asdict(self.current_code_compatibility),
        }

    def validate_immutable(self) -> None:
        """Reauthenticate every returned trainer-facing identity and payload."""
        self._validate_contents()
        if canonical_json_sha256(self._payload()) != self.artifact_evidence_sha256:
            raise ValueError("loaded artifact evidence changed after authentication")


def write_pr_history_v5_artifact(
    history: PRSceneHistory,
    chain: PRHistoryChain,
    *,
    selected_start_ordinal: int,
    selected_stop_ordinal: int,
    trajectory_id: str,
    bundle_path: str | pathlib.Path,
    source_path: str | pathlib.Path,
    bundle_uri: str,
    source_uri: str,
    expected_history_chain_sha256: str,
    expected_root_checkpoint_sha256: str,
    max_chain_transitions: int = 64,
) -> BuiltPRV5Artifact:
    """Write one complete root chain without overwriting destination files."""
    _sha256(expected_history_chain_sha256, "expected_history_chain_sha256")
    _sha256(expected_root_checkpoint_sha256, "expected_root_checkpoint_sha256")
    _positive_integer(max_chain_transitions, "max_chain_transitions")
    destination_bundle = _validate_destination(bundle_path, "bundle_path")
    destination_source = _validate_destination(source_path, "source_path")
    _validate_uri(bundle_uri, "bundle_uri")
    _validate_uri(source_uri, "source_uri")
    if destination_bundle == destination_source:
        raise ValueError("bundle_path and source_path must differ")
    snapshot = _snapshot_chain(chain)
    if snapshot.failed_reference is not None:
        raise ValueError("durable v5 training artifacts require a successful accepted range")
    if len(snapshot.transitions) > max_chain_transitions:
        raise ValueError("source chain exceeds max_chain_transitions")
    if snapshot.chain_sha256 != expected_history_chain_sha256:
        raise ValueError("source chain differs from the externally pinned SHA-256")
    canonical_history = _reconstruct_canonical_history(history)
    count = len(snapshot.transitions)
    if count < 1:
        raise ValueError("durable PR-history artifacts require at least one accepted transition")
    if (
        type(selected_start_ordinal) is not int
        or type(selected_stop_ordinal) is not int
        or selected_start_ordinal != 0
        or selected_stop_ordinal != count
    ):
        raise ValueError("durable PR-history artifacts require complete selection of the source chain")
    live_loaded_samples = tuple(
        load_pr_history_v5_sample(
            canonical_history,
            snapshot,
            snapshot.transitions[ordinal],
            trajectory_id=trajectory_id,
            expected_history_chain_sha256=expected_history_chain_sha256,
            expected_root_checkpoint_sha256=expected_root_checkpoint_sha256,
            max_chain_transitions=max_chain_transitions,
            device="cpu",
        )
        for ordinal in range(selected_start_ordinal, selected_stop_ordinal)
    )
    loaded_samples = tuple(
        load_source_bound_pr_history_v5_sample(
            canonical_history.manifest,
            canonical_history.static_bundle,
            canonical_history._base_scene,
            snapshot,
            snapshot.transitions[ordinal],
            trajectory_id=trajectory_id,
            expected_history_chain_sha256=expected_history_chain_sha256,
            expected_root_checkpoint_sha256=expected_root_checkpoint_sha256,
            max_chain_transitions=max_chain_transitions,
            device="cpu",
        )
        for ordinal in range(selected_start_ordinal, selected_stop_ordinal)
    )
    for live, source_bound in zip(live_loaded_samples, loaded_samples, strict=True):
        if (
            live.training_sample.sample_record.as_dict() != source_bound.training_sample.sample_record.as_dict()
            or live.reference_acceptance.as_dict() != source_bound.reference_acceptance.as_dict()
            or live.physical_integration.as_dict() != source_bound.physical_integration.as_dict()
        ):
            raise ValueError("current-code and source-bound loaders disagree while building the artifact")
    arrays = _collect_arrays(canonical_history, snapshot)
    bundle_sha256 = _write_bundle(destination_bundle, arrays)
    try:
        selection = _selection_record(
            snapshot,
            loaded_samples,
            selected_start_ordinal,
            selected_stop_ordinal,
        )
        ingredients = _provenance_ingredients(
            canonical_history.kind,
            canonical_history.manifest,
            canonical_history._base_scene,
            snapshot,
        )
        source = _source_payload(
            canonical_history.kind,
            canonical_history.manifest,
            canonical_history.static_bundle,
            canonical_history._base_scene,
            snapshot,
            loaded_samples,
            arrays,
            selection,
            ingredients,
            trajectory_id=trajectory_id,
            bundle_uri=bundle_uri,
            source_uri=source_uri,
            bundle_sha256=bundle_sha256,
        )
        source_sha256 = _write_source(destination_source, source)
    except Exception:
        destination_bundle.unlink(missing_ok=True)
        raise
    try:
        trajectory = _build_dataset_record(
            source,
            loaded_samples,
            source_file_sha256=source_sha256,
            bundle_file_sha256=bundle_sha256,
        )
    except Exception:
        destination_source.unlink(missing_ok=True)
        destination_bundle.unlink(missing_ok=True)
        raise
    return BuiltPRV5Artifact(
        bundle_path=destination_bundle,
        source_path=destination_source,
        bundle_file_sha256=bundle_sha256,
        source_file_sha256=source_sha256,
        source_record_sha256=str(source["source_record_sha256"]),
        loaded_samples=loaded_samples,
        trajectory=trajectory,
    )


def _read_source(
    path: pathlib.Path,
    expected_sha256: str,
    *,
    max_source_bytes: int,
) -> dict[str, object]:
    with path.open("rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("source_path must be a regular file")
        if status.st_size > max_source_bytes:
            raise ValueError("source JSON exceeds max_source_bytes")
        raw = stream.read(max_source_bytes + 1)
    if len(raw) > max_source_bytes:
        raise ValueError("source JSON exceeds max_source_bytes")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source file differs from its externally pinned SHA-256")
    try:
        source = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        canonical_bytes = _canonical_json_bytes(source) + b"\n"
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("source JSON is not canonical finite JSON") from exc
    if type(source) is not dict or raw != canonical_bytes:
        raise ValueError("source JSON bytes are not in canonical encoding")
    if set(source) != _SOURCE_TOP_LEVEL_KEYS:
        raise ValueError("source JSON top-level keys differ from the canonical contract")
    declared = source.pop("source_record_sha256", None)
    _sha256(declared, "source_record_sha256")
    actual = canonical_json_sha256(source)
    source["source_record_sha256"] = declared
    if declared != actual:
        raise ValueError("source record SHA-256 changed after authentication")
    if (
        type(source.get("schema_version")) is not int
        or source["schema_version"] != 3
        or type(source.get("contract")) is not str
        or source["contract"] != _SOURCE_CONTRACT
    ):
        raise ValueError("source JSON uses an unsupported contract")
    expected_zip = {
        "compression": "stored",
        "timestamp": list(_ZIP_TIMESTAMP),
        "entry_order": "lexicographic",
    }
    if (
        type(source.get("bundle_contract")) is not str
        or source["bundle_contract"] != _BUNDLE_CONTRACT
        or not _canonical_json_equal(source.get("npy_version"), list(_NPY_VERSION))
        or not _canonical_json_equal(source.get("zip"), expected_zip)
    ):
        raise ValueError("source JSON changed the deterministic bundle writer contract")
    return source


def _safe_npy_array(
    raw: bytes,
    declared: Mapping[str, object],
    expected_dtype: np.dtype,
    expected_rank: int,
    *,
    key: str,
    max_array_bytes: int,
) -> np.ndarray:
    if type(declared) is not dict or set(declared) != {"dtype", "shape", "sha256", "nbytes"}:
        raise ValueError(f"source array {key!r} record has noncanonical keys")
    declared_shape = declared.get("shape")
    if (
        type(declared_shape) is not list
        or len(declared_shape) != expected_rank
        or any(type(extent) is not int or extent < 0 for extent in declared_shape)
    ):
        raise ValueError(f"source array {key!r} record has a noncanonical shape")
    if type(declared.get("dtype")) is not str or type(declared.get("nbytes")) is not int:
        raise ValueError(f"source array {key!r} record has noncanonical scalar types")
    _sha256(declared.get("sha256"), f"source array {key!r} SHA-256")
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version != _NPY_VERSION:
            raise ValueError(f"bundle array {key!r} does not use NPY version {_NPY_VERSION}")
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
            stream,
            max_header_size=_MAX_NPY_HEADER_BYTES,
        )
    except (EOFError, ValueError) as exc:
        raise ValueError(f"bundle array {key!r} has an invalid bounded NPY header") from exc
    canonical_dtype = expected_dtype.newbyteorder("<")
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.str != canonical_dtype.str
        or fortran_order
    ):
        raise ValueError(f"bundle array {key!r} has a noncanonical dtype or storage order")
    if not isinstance(shape, tuple) or len(shape) != expected_rank:
        raise ValueError(f"bundle array {key!r} has a noncanonical rank")
    if canonical_dtype.itemsize < 1:
        raise ValueError(f"bundle array {key!r} has a zero-size dtype")
    element_count = 1
    for extent in shape:
        if type(extent) is not int or extent < 0:
            raise ValueError(f"bundle array {key!r} has a noncanonical shape")
        if extent > max_array_bytes // canonical_dtype.itemsize:
            raise ValueError(f"bundle array {key!r} has an oversized individual dimension")
        element_count *= extent
        if element_count > max_array_bytes:
            raise ValueError(f"bundle array {key!r} shape exceeds the configured byte bound")
    data_nbytes = element_count * canonical_dtype.itemsize
    if data_nbytes > max_array_bytes:
        raise ValueError(f"bundle array {key!r} payload exceeds the configured byte bound")
    data_offset = stream.tell()
    if data_offset > len(raw) or len(raw) - data_offset != data_nbytes:
        raise ValueError(f"bundle array {key!r} payload length differs from its bounded NPY header")
    if (
        type(declared.get("nbytes")) is not int
        or declared["nbytes"] != data_nbytes
        or declared.get("dtype") != canonical_dtype.str
        or tuple(declared_shape) != shape
    ):
        raise ValueError(f"bundle array {key!r} header differs from its source record")
    try:
        array = np.frombuffer(raw, dtype=canonical_dtype, count=element_count, offset=data_offset).reshape(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bundle array {key!r} payload cannot be decoded safely") from exc
    return _canonical_array(array)


def _streams_equal(left, right, *, chunk_size: int = 1024 * 1024) -> bool:
    left.seek(0, os.SEEK_END)
    right.seek(0, os.SEEK_END)
    if left.tell() != right.tell():
        return False
    left.seek(0)
    right.seek(0)
    while left_chunk := left.read(chunk_size):
        if left_chunk != right.read(len(left_chunk)):
            return False
    return right.read(1) == b""


def _read_bundle(
    path: pathlib.Path,
    expected_sha256: str,
    inventory: Mapping[str, object],
    expected_specs: Mapping[str, tuple[np.dtype, int]],
    *,
    max_entries: int,
    max_bundle_file_bytes: int,
    max_uncompressed_bytes: int,
) -> dict[str, np.ndarray]:
    if type(inventory) is not dict or len(inventory) > max_entries:
        raise ValueError("source array inventory exceeds max_entries")
    if set(inventory) != set(expected_specs):
        raise ValueError("source array inventory differs from the exact chain-derived inventory")
    expected_names = {f"{name}.npy" for name in inventory}
    arrays: dict[str, np.ndarray] = {}
    with path.open("rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("bundle_path must be a regular file")
        if status.st_size > max_bundle_file_bytes:
            raise ValueError("bundle file exceeds max_bundle_file_bytes")
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("bundle file differs from its externally pinned SHA-256")
        stream.seek(0)
        with zipfile.ZipFile(stream, mode="r") as archive:
            if archive.comment != b"":
                raise ValueError("bundle ZIP has a noncanonical archive comment")
            infos = archive.infolist()
            if len(infos) > max_entries or len(infos) != len(expected_names):
                raise ValueError("bundle ZIP entry count differs from the bounded source inventory")
            names = [info.filename for info in infos]
            if names != sorted(names) or len(names) != len(set(names)) or set(names) != expected_names:
                raise ValueError("bundle ZIP entries differ from the source inventory")
            total_uncompressed = 0
            for info in infos:
                name = info.filename
                if (
                    name.startswith("/")
                    or "\\" in name
                    or any(part in ("", ".", "..") for part in name.split("/"))
                    or info.date_time != _ZIP_TIMESTAMP
                    or info.create_system != 3
                    or info.external_attr != 0o600 << 16
                    or info.flag_bits != 0
                    or info.internal_attr != 0
                    or info.extra != b""
                    or info.comment != b""
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.file_size != info.compress_size
                ):
                    raise ValueError("bundle ZIP contains an unsafe or noncanonical entry")
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise ValueError("bundle ZIP exceeds max_uncompressed_bytes")
                key = name.removesuffix(".npy")
                declared = inventory[key]
                if not isinstance(declared, Mapping):
                    raise ValueError("source array inventory has an invalid size")
                raw = archive.read(info)
                canonical = _safe_npy_array(
                    raw,
                    declared,
                    expected_specs[key][0],
                    expected_specs[key][1],
                    key=key,
                    max_array_bytes=max_uncompressed_bytes,
                )
                if not _canonical_json_equal(_array_record(canonical), declared):
                    raise ValueError(f"bundle array {key!r} differs from its source record")
                if raw != _npy_bytes(canonical):
                    raise ValueError(f"bundle array {key!r} does not use the canonical NPY encoding")
                arrays[key] = canonical
        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b") as canonical_stream:
            _write_canonical_zip(canonical_stream, arrays)
            if not _streams_equal(stream, canonical_stream):
                raise ValueError("bundle file bytes differ from the complete canonical ZIP encoding")
    return arrays


def _atomic_coordinate(record: Mapping[str, object], name: str) -> AtomicCoordinate:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} coordinate record must be a mapping")
    coordinate = AtomicCoordinate(record["frame"], record["substep"])
    if not _canonical_json_equal(coordinate.as_dict(), record):
        raise ValueError(f"{name} coordinate record changed ordinal")
    return coordinate


def _reconstruct_manifest(record: object) -> PRHistoryManifest:
    if not isinstance(record, Mapping):
        raise ValueError("source history manifest must be a mapping")
    manifest = PRHistoryManifest(
        kind=str(record["kind"]),
        source_path=str(record["source_path"]),
        base_physical_sha256=str(record["base_physical_sha256"]),
        topology_sha256=str(record["topology_sha256"]),
        material_sha256=str(record["material_sha256"]),
        compression_ratio=record["compression_ratio"],
        schedule=record["schedule"],
        total_frames=record["total_frames"],
        substeps_per_frame=record["substeps_per_frame"],
        dt_seconds=record["dt_seconds"],
        schema_version=record["schema_version"],
        source_revision=str(record["source_revision"]),
    )
    if not _canonical_json_equal(manifest.as_dict(), record):
        raise ValueError("reconstructed source history manifest differs from its record")
    return manifest


def _array_at(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    try:
        return arrays[key]
    except KeyError as exc:
        raise ValueError(f"durable array inventory is missing {key!r}") from exc


def _reconstruct_static_bundle(
    manifest: PRHistoryManifest,
    record: object,
    arrays: Mapping[str, np.ndarray],
) -> PRHistoryStaticBundle:
    if not isinstance(record, Mapping):
        raise ValueError("source static bundle must be a mapping")
    expected_dtypes = {
        "rest_q": np.float32,
        "tet_indices": np.int64,
        "tet_poses": np.float32,
        "mass": np.float32,
        "tet_materials": np.float32,
        "gravity": np.float32,
        "external_force": np.float32,
    }
    values = {}
    for name, dtype in expected_dtypes.items():
        value = _array_at(arrays, f"static/{name}")
        _require_array_dtype(value, dtype, f"static/{name}")
        values[name] = value
    static = PRHistoryStaticBundle(
        manifest_sha256=manifest.manifest_sha256,
        base_physical_sha256=manifest.base_physical_sha256,
        topology_sha256=manifest.topology_sha256,
        material_sha256=manifest.material_sha256,
        **values,
    )
    if not _canonical_json_equal(static.as_dict(), record):
        raise ValueError("reconstructed source static bundle differs from its record")
    return static


def _reconstruct_base_scene(record: object, arrays: Mapping[str, np.ndarray]) -> TetBenchmarkScene:
    if not isinstance(record, Mapping) or set(record) != {
        "name",
        "source",
        "dt_seconds",
        "metadata",
        "manifest",
    }:
        raise ValueError("source base-scene record has noncanonical keys")
    integer_dtypes = {
        "tet_indices": np.int64,
        "tri_indices": np.int64,
        "particle_flags": np.int32,
        "color_group_offsets": np.int64,
        "color_group_particles": np.int64,
        "pinned_indices": np.int64,
    }
    values = {}
    for name in _BASE_SCENE_ARRAY_NAMES:
        value = _array_at(arrays, f"base_scene/{name}")
        _require_array_dtype(value, integer_dtypes.get(name, np.float64), f"base_scene/{name}")
        values[name] = value
    scene = TetBenchmarkScene(
        name=str(record["name"]),
        source=str(record["source"]),
        dt=float(record["dt_seconds"]),
        metadata=record["metadata"],
        **values,
    )
    if not _canonical_json_equal(scene.manifest(), record["manifest"]) or not _canonical_json_equal(
        _base_scene_record(scene), record
    ):
        raise ValueError("reconstructed source base scene differs from its record")
    return scene


def _reconstruct_state(
    manifest_sha256: str,
    record: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> CommittedState:
    coordinate = _atomic_coordinate(record["coordinate"], "committed state")
    prefix = _state_array_prefix(coordinate.ordinal)
    _require_array_dtype(arrays[f"{prefix}/q"], np.float32, f"{prefix}/q")
    _require_array_dtype(arrays[f"{prefix}/qd"], np.float32, f"{prefix}/qd")
    _require_array_dtype(
        arrays[f"{prefix}/particle_flags"],
        np.int32,
        f"{prefix}/particle_flags",
    )
    state_value = CommittedState(
        manifest_sha256=manifest_sha256,
        coordinate=coordinate,
        q=arrays[f"{prefix}/q"],
        qd=arrays[f"{prefix}/qd"],
        particle_flags=arrays[f"{prefix}/particle_flags"],
    )
    if not _canonical_json_equal(state_value.as_dict(), record):
        raise ValueError("reconstructed committed state differs from the source record")
    return state_value


def _reconstruct_chain(
    manifest: PRHistoryManifest,
    source_chain: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> PRHistoryChain:
    if source_chain.get("failed_reference") is not None or source_chain.get("prior_chain_sha256") is not None:
        raise ValueError("durable loader requires a successful root-origin chain")
    if source_chain.get("timing_policy") != "excluded-nondeterministic-diagnostics-v1":
        raise ValueError("source chain changed its timing exclusion policy")
    if not _canonical_json_equal(source_chain.get("manifest"), manifest.as_dict()):
        raise ValueError("source chain manifest differs from source-bound reconstruction")
    state_records = source_chain.get("accepted_states")
    transition_records = source_chain.get("transitions")
    if not isinstance(state_records, list) or not isinstance(transition_records, list):
        raise ValueError("source chain state/transition records must be lists")
    if len(state_records) != len(transition_records) + 1:
        raise ValueError("source chain accepted-state count is disconnected")
    states = tuple(_reconstruct_state(manifest.manifest_sha256, state_record, arrays) for state_record in state_records)
    initial_record = source_chain["initial_checkpoint"]
    initial = HistoryCheckpoint(
        manifest_sha256=manifest.manifest_sha256,
        state=states[0],
        prior_transition_sha256=initial_record["prior_transition_sha256"],
        prefix_sha256=initial_record["prefix_sha256"],
    )
    if not _canonical_json_equal(initial.as_dict(), initial_record):
        raise ValueError("reconstructed initial checkpoint differs from the source record")

    transitions: list[HistoryTransition] = []
    for index, (record, input_state, output_state) in enumerate(
        zip(transition_records, states[:-1], states[1:], strict=True)
    ):
        coordinate = _atomic_coordinate(record["coordinate"], f"transition {index}")
        if coordinate.ordinal != index:
            raise ValueError("durable root-chain transition ordinals are not contiguous")
        applied_record = record["applied_record"]
        prefix = _transition_array_prefix(index)
        for suffix, dtype in (
            ("applied_q", np.float32),
            ("applied_particle_flags", np.int32),
            ("pinned_indices", np.int64),
            ("pin_targets", np.float32),
            ("inertial_target", np.float32),
            ("reference_positions", np.float64),
        ):
            _require_array_dtype(arrays[f"{prefix}/{suffix}"], dtype, f"{prefix}/{suffix}")
        applied = AppliedAtomicState(
            manifest_sha256=manifest.manifest_sha256,
            coordinate=coordinate,
            input_state_sha256=str(applied_record["input_state_sha256"]),
            callback_applied=applied_record["callback_applied"],
            action=str(applied_record["action"]),
            schedule_value_name=applied_record["schedule_value_name"],
            schedule_value=applied_record["schedule_value"],
            q=arrays[f"{prefix}/applied_q"],
            particle_flags=arrays[f"{prefix}/applied_particle_flags"],
            pinned_indices=arrays[f"{prefix}/pinned_indices"],
            pin_targets=arrays[f"{prefix}/pin_targets"],
        )
        if not _canonical_json_equal(applied.as_dict(), applied_record):
            raise ValueError("reconstructed applied state differs from the source record")
        transition = HistoryTransition(
            manifest_sha256=str(record["manifest_sha256"]),
            coordinate=coordinate,
            next_coordinate=_atomic_coordinate(record["next_coordinate"], f"transition {index} next"),
            input_state_sha256=str(record["input_state_sha256"]),
            input_prefix_sha256=str(record["input_prefix_sha256"]),
            input_state=input_state,
            applied_state=applied,
            applied_record=applied_record,
            scene_sha256=str(record["scene_sha256"]),
            objective_instance_sha256=str(record["objective_instance_sha256"]),
            dt_seconds=float(record["dt_seconds"]),
            topology_sha256=str(record["topology_sha256"]),
            material_sha256=str(record["material_sha256"]),
            inertial_target=arrays[f"{prefix}/inertial_target"],
            reference_record=record["reference_record"],
            reference_positions=arrays[f"{prefix}/reference_positions"],
            output_state=output_state,
        )
        if not _canonical_json_equal(transition.as_dict(), record):
            raise ValueError("reconstructed transition differs from the source record")
        transitions.append(transition)

    final_record = source_chain["final_checkpoint"]
    final = HistoryCheckpoint(
        manifest_sha256=manifest.manifest_sha256,
        state=states[-1],
        prior_transition_sha256=final_record["prior_transition_sha256"],
        prefix_sha256=final_record["prefix_sha256"],
    )
    if not _canonical_json_equal(final.as_dict(), final_record):
        raise ValueError("reconstructed final checkpoint differs from the source record")
    timings = tuple(
        TransitionTiming(
            coordinate=transition.coordinate,
            accepted=True,
            values={"durable_reconstruction": "source-timings-excluded"},
        )
        for transition in transitions
    )
    chain = PRHistoryChain(
        manifest=manifest,
        initial_checkpoint=initial,
        transitions=tuple(transitions),
        timings=timings,
        final_checkpoint=final,
        termination=str(source_chain["termination"]),
    )
    actual = _chain_record_without_timings(chain)
    if not _canonical_json_equal(actual, source_chain):
        raise ValueError("reconstructed PR history chain differs from the source record")
    return chain


def _current_code_compatibility(history: SourceBoundPRHistory) -> CurrentCodeCompatibility:
    try:
        current = PRSceneHistory(history.kind)
        manifest_matches = current.manifest.as_dict() == history.manifest.as_dict()
        static_matches = current.static_bundle.as_dict() == history.static_bundle.as_dict()
        root_matches = current.initial_checkpoint.as_dict() == history.initial_checkpoint.as_dict()
        base_matches = current._base_scene.manifest() == history.base_scene.manifest()
        compatible = manifest_matches and static_matches and root_matches and base_matches
        return CurrentCodeCompatibility(
            compatible=compatible,
            manifest_matches=manifest_matches,
            static_bundle_matches=static_matches,
            root_checkpoint_matches=root_matches,
            base_scene_matches=base_matches,
            current_manifest_sha256=current.manifest.manifest_sha256,
            current_root_checkpoint_sha256=current.initial_checkpoint.checkpoint_sha256,
            reason=None if compatible else "current checkout does not exactly reproduce the archived source identity",
        )
    except Exception as exc:
        return CurrentCodeCompatibility(
            compatible=False,
            manifest_matches=False,
            static_bundle_matches=False,
            root_checkpoint_matches=False,
            base_scene_matches=False,
            current_manifest_sha256=None,
            current_root_checkpoint_sha256=None,
            reason=f"{type(exc).__name__}: {exc}",
        )


def load_pr_history_v5_artifact(
    source_path: str | pathlib.Path,
    bundle_path: str | pathlib.Path,
    *,
    expected_source_file_sha256: str,
    expected_bundle_file_sha256: str,
    expected_history_chain_sha256: str,
    expected_root_checkpoint_sha256: str,
    device: str | torch.device = "cpu",
    max_chain_transitions: int = 64,
    max_entries: int = 2048,
    max_source_bytes: int = 64 * 1024 * 1024,
    max_bundle_file_bytes: int = 2 * 1024 * 1024 * 1024,
    max_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024,
) -> LoadedPRV5Artifact:
    """Verify deterministic files, rebuild their chain, and rerun trust checks."""
    for name, digest in (
        ("expected_source_file_sha256", expected_source_file_sha256),
        ("expected_bundle_file_sha256", expected_bundle_file_sha256),
        ("expected_history_chain_sha256", expected_history_chain_sha256),
        ("expected_root_checkpoint_sha256", expected_root_checkpoint_sha256),
    ):
        _sha256(digest, name)
    for name, value in (
        ("max_chain_transitions", max_chain_transitions),
        ("max_entries", max_entries),
        ("max_source_bytes", max_source_bytes),
        ("max_bundle_file_bytes", max_bundle_file_bytes),
        ("max_uncompressed_bytes", max_uncompressed_bytes),
    ):
        _positive_integer(value, name)
    if torch.device(device).type != "cpu":
        raise ValueError("durable artifact authentication currently requires canonical CPU reconstruction")
    source_file = pathlib.Path(source_path).resolve()
    bundle_file = pathlib.Path(bundle_path).resolve()
    source = _read_source(source_file, expected_source_file_sha256, max_source_bytes=max_source_bytes)
    artifact_uris = source.get("artifact_uris")
    if type(artifact_uris) is not dict or set(artifact_uris) != {"bundle", "source"}:
        raise ValueError("source artifact_uris record has noncanonical keys")
    _validate_uri(artifact_uris["bundle"], "source artifact bundle URI")
    _validate_uri(artifact_uris["source"], "source artifact source URI")
    trajectory_id = source.get("trajectory_id")
    if type(trajectory_id) is not str:
        raise ValueError("source trajectory_id must be a canonical string")
    persisted_samples = source.get("samples")
    if type(persisted_samples) is not list or any(
        type(item) is not dict or set(item) != _SAMPLE_WRAPPER_KEYS for item in persisted_samples
    ):
        raise ValueError("source persisted-sample wrappers have noncanonical keys")
    source_bundle_sha256 = _sha256(source.get("bundle_sha256"), "source bundle_sha256")
    if source_bundle_sha256 != expected_bundle_file_sha256:
        raise ValueError("source JSON bundle SHA-256 differs from the external anchor")
    inventory = source.get("arrays")
    if not isinstance(inventory, dict):
        raise ValueError("source JSON array inventory must be a mapping")
    source_chain = source.get("chain")
    if not isinstance(source_chain, dict):
        raise ValueError("source chain record must be a mapping")
    transition_records = source_chain.get("transitions")
    state_records = source_chain.get("accepted_states")
    if not isinstance(transition_records, list) or not isinstance(state_records, list):
        raise ValueError("source chain transition/state inventories must be lists")
    transition_count = len(transition_records)
    if transition_count < 1 or transition_count > max_chain_transitions:
        raise ValueError("source chain transition count lies outside the configured bounds")
    if len(state_records) != transition_count + 1:
        raise ValueError("source chain accepted-state count is disconnected")
    selection = source.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("source selection record must be a mapping")
    selection_payload = dict(selection)
    declared_selection_sha256 = selection_payload.pop("selection_sha256", None)
    _sha256(declared_selection_sha256, "source selection_sha256")
    if declared_selection_sha256 != canonical_json_sha256(selection_payload):
        raise ValueError("source contiguous-selection record changed after authentication")
    integer_selection_fields = (
        "source_transition_count",
        "selected_start_ordinal",
        "selected_stop_ordinal_exclusive",
        "excluded_prefix_count",
        "excluded_suffix_count",
    )
    if (
        any(type(selection.get(name)) is not int for name in integer_selection_fields)
        or selection.get("source_transition_count") != transition_count
        or selection.get("selected_start_ordinal") != 0
        or selection.get("selected_stop_ordinal_exclusive") != transition_count
        or selection.get("excluded_prefix_count") != 0
        or selection.get("excluded_suffix_count") != 0
        or len(persisted_samples) != transition_count
    ):
        raise ValueError("durable PR-history artifact source must select the complete chain")
    expected_specs = _expected_array_specs(transition_count)
    arrays = _read_bundle(
        bundle_file,
        expected_bundle_file_sha256,
        inventory,
        expected_specs,
        max_entries=max_entries,
        max_bundle_file_bytes=max_bundle_file_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    manifest = _reconstruct_manifest(source.get("history_manifest"))
    if type(source.get("history_kind")) is not str or source["history_kind"] != manifest.kind:
        raise ValueError("source history kind differs from its source-bound manifest")
    static_bundle = _reconstruct_static_bundle(manifest, source.get("static_bundle"), arrays)
    base_scene = _reconstruct_base_scene(source.get("base_scene"), arrays)
    chain = _reconstruct_chain(manifest, source_chain, arrays)
    if chain.chain_sha256 != expected_history_chain_sha256:
        raise ValueError("reconstructed chain differs from the external anchor")
    if chain.initial_checkpoint.checkpoint_sha256 != expected_root_checkpoint_sha256:
        raise ValueError("reconstructed source-bound root differs from the external anchor")
    if len(chain.transitions) > max_chain_transitions:
        raise ValueError("reconstructed chain exceeds max_chain_transitions")
    history = SourceBoundPRHistory(
        kind=manifest.kind,
        manifest=manifest,
        static_bundle=static_bundle,
        initial_checkpoint=chain.initial_checkpoint,
        base_scene=base_scene,
    )
    canonical_ingredients = _provenance_ingredients(
        history.kind,
        history.manifest,
        history.base_scene,
        chain,
    )
    if not _canonical_json_equal(source.get("provenance_ingredients"), canonical_ingredients):
        raise ValueError("persisted provenance ingredients differ from canonical reconstruction")
    if not _canonical_json_equal(source.get("trust_scope"), _trust_scope_record()):
        raise ValueError("source JSON changed the durable trust scope")

    selection = source["selection"]
    start = selection.get("selected_start_ordinal")
    stop = selection.get("selected_stop_ordinal_exclusive")
    if start != 0 or stop != len(chain.transitions):
        raise ValueError("durable PR-history artifact source must select the complete chain")
    if len(persisted_samples) != stop - start:
        raise ValueError("source persisted-sample count differs from its selection")
    loaded_samples = []
    for persisted, ordinal in zip(persisted_samples, range(start, stop), strict=True):
        if not isinstance(persisted, dict):
            raise ValueError("persisted sample evidence must be a mapping")
        loaded = load_source_bound_pr_history_v5_sample(
            history.manifest,
            history.static_bundle,
            history.base_scene,
            chain,
            chain.transitions[ordinal],
            trajectory_id=trajectory_id,
            expected_history_chain_sha256=expected_history_chain_sha256,
            expected_root_checkpoint_sha256=expected_root_checkpoint_sha256,
            max_chain_transitions=max_chain_transitions,
            device="cpu",
        )
        if not _canonical_json_equal(loaded.training_sample.sample_record.as_dict(), persisted.get("sample_record")):
            raise ValueError("persisted sample record differs from trusted reconstruction")
        if not _canonical_json_equal(loaded.reference_acceptance.as_dict(), persisted.get("reference_acceptance")):
            raise ValueError("persisted reference acceptance differs from independent reconstruction")
        if not _canonical_json_equal(loaded.physical_integration.as_dict(), persisted.get("physical_integration")):
            raise ValueError("persisted physical integration differs from independent reconstruction")
        if not _canonical_json_equal(_operator_geometry_record(loaded), persisted.get("operator_geometry")):
            raise ValueError("persisted operator geometry differs from trusted reconstruction")
        if loaded.loaded_sample_sha256 != persisted.get("loaded_sample_sha256"):
            raise ValueError("persisted loaded-sample SHA-256 differs from trusted reconstruction")
        loaded_samples.append(loaded)
    canonical_selection = _selection_record(chain, loaded_samples, start, stop)
    if not _canonical_json_equal(canonical_selection, selection):
        raise ValueError("persisted contiguous selection differs from trusted reconstruction")
    canonical_source = _source_payload(
        history.kind,
        history.manifest,
        history.static_bundle,
        history.base_scene,
        chain,
        loaded_samples,
        arrays,
        canonical_selection,
        canonical_ingredients,
        trajectory_id=trajectory_id,
        bundle_uri=artifact_uris["bundle"],
        source_uri=artifact_uris["source"],
        bundle_sha256=expected_bundle_file_sha256,
    )
    if not _canonical_json_equal(source, canonical_source):
        raise ValueError("source JSON differs from the complete trusted reconstruction")
    trajectory = _build_dataset_record(
        source,
        loaded_samples,
        source_file_sha256=expected_source_file_sha256,
        bundle_file_sha256=expected_bundle_file_sha256,
    )
    return LoadedPRV5Artifact(
        bundle_path=bundle_file,
        source_path=source_file,
        bundle_file_sha256=expected_bundle_file_sha256,
        source_file_sha256=expected_source_file_sha256,
        source_record_sha256=str(source["source_record_sha256"]),
        history_manifest_sha256=history.manifest.manifest_sha256,
        root_checkpoint_sha256=history.initial_checkpoint.checkpoint_sha256,
        source_chain_sha256=chain.chain_sha256,
        static_bundle_sha256=history.static_bundle.static_sha256,
        base_scene_sha256=str(history.base_scene.manifest()["scene_sha256"]),
        loaded_samples=tuple(loaded_samples),
        trajectory=trajectory,
        current_code_compatibility=_current_code_compatibility(history),
    )
