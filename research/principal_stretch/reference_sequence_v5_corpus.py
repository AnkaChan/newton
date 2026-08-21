# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build canonical v5 split metadata from authenticated reference sequences.

This is a dataset-preparation boundary, not a training payload cache.  It
materializes one transition at a time through :class:`ReferenceSequenceV5Bridge`,
retains only immutable sample records and transition-key bindings, and drops
each ``V5TrainingSample`` before advancing to the next transition.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .reference_sequence_dataset import (
    REFERENCE_EXECUTION_DT_FLOAT32_BITS,
    REFERENCE_EXECUTION_DT_SECONDS,
    REFERENCE_REQUESTED_DT_SECONDS,
    ReferenceSequenceDataset,
    ReferenceSequenceRecord,
    ReferenceTransitionKey,
    canonical_reference_state_float64_sha256,
    reference_sequence_index_header,
)
from .reference_sequence_v5_bridge import ReferenceSequenceV5Bridge
from .v5_checkpoint import canonical_json_sha256
from .v5_dataset import (
    DatasetRole,
    ReferenceSequenceProvenance,
    SplitManifest,
    TrajectoryRecord,
    TrajectorySampleRecord,
    _verify_manifest,
    verify_file_sha256,
)

_CORPUS_CONTRACT = "pss-reference-sequence-v5-corpus-v1"
_SOURCE_CHAIN_CONTRACT = "pss-reference-sequence-source-chain-v1"
_LOAD_PROGRAM_CONTRACT = "pss-reference-sequence-dynamics-program-v1"
_PRODUCER_INDEX_SCHEMA = "pss-free-body-reference-index-v1"
_PRODUCER_SHARD_SCHEMA = "pss-free-body-reference-shard-v1"
_SPLIT_INDEX_FILENAME = "reference-sequence-split-index.json"
_PRODUCER_INDEX_KEYS = frozenset(
    (
        "schema",
        "protocol",
        "protocol_sha256",
        "base_seed",
        "samples_per_asset",
        "hierarchy_config",
        "asset_count",
        "accepted_sequence_count",
        "assets",
    )
)
_PRODUCER_ASSET_KEYS = frozenset(
    (
        "asset_id",
        "source",
        "source_sha256",
        "vertex_count",
        "tet_count",
        "static_npz",
        "identities",
        "sequences",
    )
)
_PRODUCER_SEQUENCE_KEYS = frozenset(("sequence_id", "deformation_seed", "velocity_seed", "manifest", "sequence_npz"))
_PRODUCER_MANIFEST_KEYS = frozenset(
    (
        "schema",
        "asset_id",
        "source",
        "source_sha256",
        "sequence_id",
        "deformation_seed",
        "velocity_seed",
        "reference_accepted",
        "step_count",
        "protocol",
        "identities",
        "initial_scene",
        "initial_scene_sha256",
        "normalization",
        "files",
        "inventory_sha256",
    )
)


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_object(path: pathlib.Path, name: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_json_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read canonical {name} {path}") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} root must be a JSON object")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _relative_artifact_path(
    producer_root: pathlib.Path,
    base: pathlib.Path,
    value: object,
    name: str,
) -> tuple[str, pathlib.Path]:
    relative = _identifier(value, name)
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or relative != pure.as_posix()
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in relative
    ):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    resolved = (base / pathlib.Path(*pure.parts)).resolve()
    try:
        root_relative = resolved.relative_to(producer_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{name} resolves outside the producer directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} does not name an existing file")
    return root_relative, resolved


def _nested_artifact(
    producer_root: pathlib.Path,
    value: object,
    name: str,
    *,
    verify: bool = True,
) -> tuple[str, pathlib.Path, str]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain path and sha256 exactly")
    relative, path = _relative_artifact_path(producer_root, producer_root, value["path"], f"{name} path")
    digest = _sha256(value["sha256"], f"{name} sha256")
    if verify:
        verify_file_sha256(path, digest)
    return relative, path, digest


def _manifest_artifact(
    producer_root: pathlib.Path,
    manifest_path: pathlib.Path,
    value: object,
    name: str,
    *,
    arrays: bool,
    verify: bool = True,
) -> tuple[str, pathlib.Path, str, Mapping[str, object] | None]:
    expected_keys = {"path", "bytes", "sha256"} | ({"arrays"} if arrays else set())
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError(f"producer manifest {name} has an unexpected inventory")
    relative, path = _relative_artifact_path(
        producer_root,
        manifest_path.parent,
        value["path"],
        f"producer manifest {name} path",
    )
    byte_count = value["bytes"]
    if type(byte_count) is not int or byte_count < 0 or path.stat().st_size != byte_count:
        raise ValueError(f"producer manifest {name} byte count differs from its file")
    digest = _sha256(value["sha256"], f"producer manifest {name} sha256")
    if verify:
        verify_file_sha256(path, digest)
    inventory = value.get("arrays")
    if arrays and type(inventory) is not dict:
        raise ValueError(f"producer manifest {name} arrays must be a JSON object")
    return relative, path, digest, inventory


@dataclasses.dataclass(frozen=True)
class ReferenceSequenceSplitIndexBuild:
    """Deterministic build receipt and exact bytes for one flat split index."""

    producer_index_path: pathlib.Path
    producer_index_file_sha256: str
    split_index_path: pathlib.Path
    split_index_bytes: bytes = dataclasses.field(repr=False)
    split_index_file_sha256: str
    dataset_index_sha256: str
    asset_count: int
    sequence_count: int

    def __post_init__(self) -> None:
        if not self.producer_index_path.is_absolute() or not self.split_index_path.is_absolute():
            raise ValueError("split-index build paths must be absolute")
        for name in (
            "producer_index_file_sha256",
            "split_index_file_sha256",
            "dataset_index_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.split_index_bytes) is not bytes:
            raise TypeError("split_index_bytes must be immutable bytes")
        if hashlib.sha256(self.split_index_bytes).hexdigest() != self.split_index_file_sha256:
            raise ValueError("split_index_file_sha256 differs from split_index_bytes")
        for name in ("asset_count", "sequence_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_value, right_value) for left_value, right_value in zip(left, right, strict=True)
        )
    return left == right


def _canonical_asset_roles(
    asset_roles: Mapping[str, DatasetRole | str],
    producer_asset_ids: set[str],
) -> dict[str, DatasetRole]:
    if isinstance(asset_roles, (str, bytes)) or not isinstance(asset_roles, Mapping):
        raise TypeError("asset_roles must be a mapping from asset_id to DatasetRole")
    result: dict[str, DatasetRole] = {}
    for asset_id, role in asset_roles.items():
        canonical_asset_id = _identifier(asset_id, "asset_roles asset_id")
        try:
            result[canonical_asset_id] = DatasetRole(role)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"asset_roles[{canonical_asset_id!r}] is not a registered dataset role") from exc
    if set(result) != producer_asset_ids:
        raise ValueError("asset_roles must exactly cover producer assets without missing or extra entries")
    return result


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _flat_sequence_record(
    *,
    producer_root: pathlib.Path,
    root_protocol: Mapping[str, object],
    root_protocol_sha256: str,
    asset: Mapping[str, object],
    sequence: Mapping[str, object],
    role: DatasetRole,
) -> dict[str, object]:
    asset_id = _identifier(asset["asset_id"], "producer asset_id")
    source = _identifier(asset["source"], f"producer asset {asset_id} source")
    source_sha256 = _sha256(asset["source_sha256"], f"producer asset {asset_id} source_sha256")
    vertex_count = _positive_integer(asset["vertex_count"], f"producer asset {asset_id} vertex_count")
    _positive_integer(asset["tet_count"], f"producer asset {asset_id} tet_count")
    identities = asset["identities"]
    if type(identities) is not dict:
        raise ValueError(f"producer asset {asset_id} identities must be a JSON object")
    for name in ("topology_sha256", "operator_sha256", "material_sha256", "protocol_sha256"):
        _sha256(identities.get(name), f"producer asset {asset_id} identities.{name}")
    if identities["protocol_sha256"] != root_protocol_sha256:
        raise ValueError(f"producer asset {asset_id} protocol identity differs from the producer index")

    static_relative, static_path, static_sha256 = _nested_artifact(
        producer_root,
        asset["static_npz"],
        f"producer asset {asset_id} static_npz",
        verify=False,
    )
    sequence_id = _identifier(sequence["sequence_id"], f"producer asset {asset_id} sequence_id")
    for name in ("deformation_seed", "velocity_seed"):
        value = sequence[name]
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"producer sequence {asset_id}/{sequence_id} {name} must be an integer in [0, 2**32)")
    manifest_relative, manifest_path, manifest_sha256 = _nested_artifact(
        producer_root,
        sequence["manifest"],
        f"producer sequence {asset_id}/{sequence_id} manifest",
    )
    nested_sequence_relative, nested_sequence_path, nested_sequence_sha256 = _nested_artifact(
        producer_root,
        sequence["sequence_npz"],
        f"producer sequence {asset_id}/{sequence_id} sequence_npz",
        verify=False,
    )

    manifest = _read_json_object(manifest_path, f"producer manifest {asset_id}/{sequence_id}")
    if set(manifest) != _PRODUCER_MANIFEST_KEYS or manifest.get("schema") != _PRODUCER_SHARD_SCHEMA:
        raise ValueError(f"producer manifest {asset_id}/{sequence_id} has an unregistered inventory")
    if (
        manifest["asset_id"] != asset_id
        or manifest["source"] != source
        or manifest["source_sha256"] != source_sha256
        or manifest["sequence_id"] != sequence_id
        or manifest["deformation_seed"] != sequence["deformation_seed"]
        or manifest["velocity_seed"] != sequence["velocity_seed"]
        or manifest["reference_accepted"] is not True
    ):
        raise ValueError(f"producer manifest {asset_id}/{sequence_id} identity differs from its nested index")
    if not _exact_json_equal(manifest["protocol"], root_protocol):
        raise ValueError(f"producer manifest {asset_id}/{sequence_id} protocol differs from its nested index")
    if not _exact_json_equal(manifest["identities"], identities):
        raise ValueError(f"producer manifest {asset_id}/{sequence_id} identities differ from its asset record")

    files = manifest["files"]
    if type(files) is not dict or set(files) != {"static_npz", "sequence_npz", "evidence_json"}:
        raise ValueError(f"producer manifest {asset_id}/{sequence_id} files inventory is not closed")
    manifest_static_relative, manifest_static_path, manifest_static_sha256, _static_arrays = _manifest_artifact(
        producer_root,
        manifest_path,
        files["static_npz"],
        "static_npz",
        arrays=True,
        verify=False,
    )
    manifest_sequence_relative, manifest_sequence_path, manifest_sequence_sha256, sequence_arrays = _manifest_artifact(
        producer_root,
        manifest_path,
        files["sequence_npz"],
        "sequence_npz",
        arrays=True,
        verify=False,
    )
    _manifest_artifact(
        producer_root,
        manifest_path,
        files["evidence_json"],
        "evidence_json",
        arrays=False,
    )
    if manifest_static_path != static_path or manifest_static_relative != static_relative:
        raise ValueError(f"producer asset {asset_id} static_npz path differs from its manifest")
    if manifest_static_sha256 != static_sha256:
        raise ValueError(f"producer asset {asset_id} static_npz SHA-256 differs from its manifest")
    if manifest_sequence_path != nested_sequence_path or manifest_sequence_relative != nested_sequence_relative:
        raise ValueError(f"producer sequence {asset_id}/{sequence_id} sequence_npz path differs from its manifest")
    if manifest_sequence_sha256 != nested_sequence_sha256:
        raise ValueError(f"producer sequence {asset_id}/{sequence_id} sequence_npz SHA-256 differs from its manifest")
    verify_file_sha256(static_path, static_sha256)
    verify_file_sha256(nested_sequence_path, nested_sequence_sha256)

    step_count = _positive_integer(
        manifest["step_count"],
        f"producer manifest {asset_id}/{sequence_id} step_count",
    )
    if root_protocol.get("rollout_steps") != step_count:
        raise ValueError(f"producer manifest {asset_id}/{sequence_id} step count differs from the root protocol")
    if not isinstance(sequence_arrays, Mapping) or "q" not in sequence_arrays:
        raise ValueError(f"producer manifest {asset_id}/{sequence_id} sequence inventory is missing q")
    try:
        with np.load(nested_sequence_path, allow_pickle=False) as archive:
            if "q" not in archive.files:
                raise ValueError(f"producer sequence {asset_id}/{sequence_id} is missing q")
            q = np.asarray(archive["q"])
            if q.dtype != np.dtype(np.float64) or q.shape != (step_count + 1, vertex_count, 3):
                raise ValueError(
                    f"producer sequence {asset_id}/{sequence_id} reference q must be float64 "
                    f"with shape {(step_count + 1, vertex_count, 3)}"
                )
            if not np.isfinite(q).all():
                raise ValueError(
                    f"producer sequence {asset_id}/{sequence_id} reference q must contain only finite values"
                )
            reference_hashes = [
                canonical_reference_state_float64_sha256(q[step_id + 1]) for step_id in range(step_count)
            ]
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "producer sequence" in str(exc):
            raise
        raise ValueError(f"could not read producer sequence NPZ {asset_id}/{sequence_id}") from exc

    return {
        "role": role.value,
        "asset_id": asset_id,
        "asset_source_sha256": source_sha256,
        "sequence_id": sequence_id,
        "topology_sha256": identities["topology_sha256"],
        "operator_sha256": identities["operator_sha256"],
        "material_sha256": identities["material_sha256"],
        "protocol_sha256": identities["protocol_sha256"],
        "producer_manifest_json": manifest_relative,
        "producer_manifest_json_sha256": manifest_sha256,
        "step_ids": list(range(step_count)),
        "reference_state_float64_sha256": reference_hashes,
    }


def _verify_materialized_split_index(build: ReferenceSequenceSplitIndexBuild, path: pathlib.Path) -> None:
    if path.read_bytes() != build.split_index_bytes:
        raise ValueError("materialized split index bytes differ from the deterministic build")
    dataset = ReferenceSequenceDataset.load(path)
    if dataset.index_sha256 != build.dataset_index_sha256:
        raise ValueError("materialized split index canonical identity differs from the deterministic build")


def _validate_split_index_build(build: ReferenceSequenceSplitIndexBuild) -> None:
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{build.split_index_path.name}.build-validation.",
            suffix=".tmp",
            dir=build.producer_index_path.parent,
            delete=False,
        ) as stream:
            temporary_path = pathlib.Path(stream.name)
            stream.write(build.split_index_bytes)
            stream.flush()
        _verify_materialized_split_index(build, temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_reference_sequence_split_index(
    producer_index_path: str | pathlib.Path,
    *,
    asset_roles: Mapping[str, DatasetRole | str],
) -> ReferenceSequenceSplitIndexBuild:
    """Build exact flat split-index bytes from one nested producer index."""
    producer_path = pathlib.Path(producer_index_path).resolve()
    if not producer_path.is_file():
        raise ValueError("producer_index_path must name an existing file")
    producer_root = producer_path.parent
    split_path = producer_root / _SPLIT_INDEX_FILENAME
    if producer_path == split_path:
        raise ValueError("producer index and flat split index must be different files")
    producer_payload = _read_json_object(producer_path, "producer index")
    if set(producer_payload) != _PRODUCER_INDEX_KEYS or producer_payload.get("schema") != _PRODUCER_INDEX_SCHEMA:
        raise ValueError("producer index has an unregistered root inventory")

    for name in ("base_seed",):
        value = producer_payload[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"producer index {name} must be a non-negative integer")
    samples_per_asset = _positive_integer(producer_payload["samples_per_asset"], "producer samples_per_asset")
    asset_count = _positive_integer(producer_payload["asset_count"], "producer asset_count")
    accepted_sequence_count = _positive_integer(
        producer_payload["accepted_sequence_count"],
        "producer accepted_sequence_count",
    )
    hierarchy_config = producer_payload["hierarchy_config"]
    if type(hierarchy_config) is not dict or set(hierarchy_config) != {"n_levels", "cluster_size"}:
        raise ValueError("producer hierarchy_config must contain n_levels and cluster_size exactly")
    _positive_integer(hierarchy_config["n_levels"], "producer hierarchy n_levels")
    _positive_integer(hierarchy_config["cluster_size"], "producer hierarchy cluster_size")

    protocol = producer_payload["protocol"]
    if type(protocol) is not dict:
        raise ValueError("producer protocol must be a JSON object")
    protocol_sha256 = _sha256(producer_payload["protocol_sha256"], "producer protocol_sha256")
    if canonical_json_sha256(protocol) != protocol_sha256:
        raise ValueError("producer protocol_sha256 differs from its canonical protocol")
    expected_dt = {
        "requested_dt_seconds": REFERENCE_REQUESTED_DT_SECONDS,
        "execution_dt_seconds": float(REFERENCE_EXECUTION_DT_SECONDS),
        "execution_dt_float32_bits": REFERENCE_EXECUTION_DT_FLOAT32_BITS,
    }
    if any(not _exact_json_equal(protocol.get(name), expected) for name, expected in expected_dt.items()):
        raise ValueError("producer protocol timestep differs from the registered sequence contract")

    assets = producer_payload["assets"]
    if type(assets) is not list or len(assets) != asset_count:
        raise ValueError("producer assets list differs from asset_count")
    if any(type(asset) is not dict or set(asset) != _PRODUCER_ASSET_KEYS for asset in assets):
        raise ValueError("producer asset has an unexpected inventory")
    asset_ids = [_identifier(asset["asset_id"], "producer asset_id") for asset in assets]
    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("producer asset_id values must be unique")
    roles = _canonical_asset_roles(asset_roles, set(asset_ids))

    flat_records: dict[DatasetRole, list[dict[str, object]]] = {role: [] for role in DatasetRole}
    actual_sequence_count = 0
    for asset in sorted(assets, key=lambda value: str(value["asset_id"])):
        asset_id = str(asset["asset_id"])
        sequences = asset["sequences"]
        if type(sequences) is not list or len(sequences) != samples_per_asset:
            raise ValueError(f"producer asset {asset_id} sequences differ from samples_per_asset")
        if any(type(sequence) is not dict or set(sequence) != _PRODUCER_SEQUENCE_KEYS for sequence in sequences):
            raise ValueError(f"producer asset {asset_id} sequence has an unexpected inventory")
        sequence_ids = [
            _identifier(sequence["sequence_id"], f"producer asset {asset_id} sequence_id") for sequence in sequences
        ]
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(f"producer asset {asset_id} sequence_id values must be unique")
        for sequence in sorted(sequences, key=lambda value: str(value["sequence_id"])):
            flat_records[roles[asset_id]].append(
                _flat_sequence_record(
                    producer_root=producer_root,
                    root_protocol=protocol,
                    root_protocol_sha256=protocol_sha256,
                    asset=asset,
                    sequence=sequence,
                    role=roles[asset_id],
                )
            )
            actual_sequence_count += 1
    if actual_sequence_count != accepted_sequence_count:
        raise ValueError("producer sequence inventory differs from accepted_sequence_count")

    flat_payload = reference_sequence_index_header()
    flat_payload["splits"] = {
        role.value: sorted(
            flat_records[role],
            key=lambda record: (str(record["asset_id"]), str(record["sequence_id"])),
        )
        for role in DatasetRole
    }
    split_bytes = _canonical_json_bytes(flat_payload)
    build = ReferenceSequenceSplitIndexBuild(
        producer_index_path=producer_path,
        producer_index_file_sha256=_file_sha256(producer_path),
        split_index_path=split_path,
        split_index_bytes=split_bytes,
        split_index_file_sha256=hashlib.sha256(split_bytes).hexdigest(),
        dataset_index_sha256=canonical_json_sha256(flat_payload),
        asset_count=asset_count,
        sequence_count=actual_sequence_count,
    )
    _validate_split_index_build(build)
    return build


def write_reference_sequence_split_index(
    producer_index_path: str | pathlib.Path,
    *,
    asset_roles: Mapping[str, DatasetRole | str],
) -> ReferenceSequenceSplitIndexBuild:
    """Validate and atomically publish the flat index beside its producer."""
    build = build_reference_sequence_split_index(producer_index_path, asset_roles=asset_roles)
    destination = build.split_index_path
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != build.split_index_bytes:
            raise FileExistsError(f"refusing to overwrite non-identical split index: {destination}")
        _verify_materialized_split_index(build, destination)
        return build

    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = pathlib.Path(stream.name)
            stream.write(build.split_index_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_materialized_split_index(build, temporary_path)
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != build.split_index_bytes:
                raise FileExistsError(f"refusing to overwrite non-identical split index: {destination}") from None
        _verify_materialized_split_index(build, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return build


@dataclasses.dataclass(frozen=True)
class ReferenceSequenceV5Corpus:
    """Frozen v5 manifest and lazy bridge lookup for one sequence index."""

    split_manifest: SplitManifest
    transition_keys_by_sample: Mapping[tuple[str, str], ReferenceTransitionKey]
    source_index_sha256: str
    corpus_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.split_manifest) is not SplitManifest:
            raise TypeError("split_manifest must be a canonical SplitManifest")
        _verify_manifest(self.split_manifest)
        if (
            type(self.source_index_sha256) is not str
            or len(self.source_index_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_index_sha256)
        ):
            raise ValueError("source_index_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.transition_keys_by_sample, Mapping):
            raise TypeError("transition_keys_by_sample must be a mapping")

        bindings = dict(self.transition_keys_by_sample)
        expected: dict[tuple[str, str], ReferenceTransitionKey] = {}
        for role in DatasetRole:
            for trajectory in self.split_manifest.records(role):
                provenance = trajectory.provenance
                if type(provenance) is not ReferenceSequenceProvenance:
                    raise ValueError("reference-sequence corpus trajectories require sequence-native provenance")
                if provenance.dataset_index_sha256 != self.source_index_sha256:
                    raise ValueError("trajectory provenance binds a different source index")
                for sample in trajectory.samples:
                    expected[(trajectory.trajectory_id, sample.sample_id)] = ReferenceTransitionKey(
                        asset_id=provenance.asset_id,
                        sequence_id=provenance.sequence_id,
                        step_id=sample.ordinal,
                    )
        if bindings != expected:
            raise ValueError("transition-key bindings differ from the complete split manifest")
        if any(type(key) is not tuple or len(key) != 2 for key in bindings):
            raise ValueError("transition-key lookup keys must be exact (trajectory_id, sample_id) tuples")
        if any(type(value) is not ReferenceTransitionKey for value in bindings.values()):
            raise ValueError("transition-key lookup values must be canonical ReferenceTransitionKey values")

        ordered = dict(sorted(bindings.items()))
        object.__setattr__(self, "transition_keys_by_sample", MappingProxyType(ordered))
        object.__setattr__(self, "corpus_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _CORPUS_CONTRACT,
            "source_index_sha256": self.source_index_sha256,
            "split_manifest": self.split_manifest.as_dict(),
            "transition_keys": [
                {
                    "trajectory_id": trajectory_id,
                    "sample_id": sample_id,
                    "asset_id": key.asset_id,
                    "sequence_id": key.sequence_id,
                    "step_id": key.step_id,
                }
                for (trajectory_id, sample_id), key in self.transition_keys_by_sample.items()
            ],
        }

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON-compatible corpus metadata."""
        payload = self._payload()
        payload["corpus_sha256"] = self.corpus_sha256
        return payload


def _trajectory_id(asset_id: str, sequence_id: str) -> str:
    return f"reference-sequence:{asset_id}:{sequence_id}"


def _load_program_payload(provenance: ReferenceSequenceProvenance) -> dict[str, object]:
    return {
        "contract": _LOAD_PROGRAM_CONTRACT,
        "asset_source_sha256": provenance.asset_source_sha256,
        "protocol_sha256": provenance.protocol_sha256,
        "initial_position_sha256": provenance.initial_position_sha256,
        "initial_velocity_field_sha256": provenance.initial_velocity_field_sha256,
        "deformation_seed": provenance.deformation_seed,
        "velocity_seed": provenance.velocity_seed,
        "source_transition_count": provenance.source_transition_count,
    }


def _source_chain_payload(provenance: ReferenceSequenceProvenance) -> dict[str, object]:
    return {
        "contract": _SOURCE_CHAIN_CONTRACT,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "producer_manifest_sha256": provenance.producer_manifest_sha256,
        "static_bundle_sha256": provenance.static_bundle_sha256,
        "sequence_bundle_sha256": provenance.sequence_bundle_sha256,
        "evidence_sha256": provenance.evidence_sha256,
    }


def _build_trajectory(
    dataset: ReferenceSequenceDataset,
    bridge: ReferenceSequenceV5Bridge,
    record: ReferenceSequenceRecord,
    transition_bindings: dict[tuple[str, str], ReferenceTransitionKey],
) -> TrajectoryRecord:
    anchor = dataset.provenance_anchor(record)
    provenance = ReferenceSequenceProvenance(**dataclasses.asdict(anchor))
    trajectory_id = _trajectory_id(record.asset_id, record.sequence_id)
    sample_records: list[TrajectorySampleRecord] = []
    topology_sha256: str | None = None
    operator_geometry_sha256: str | None = None
    material_sha256: str | None = None
    for step_id in record.step_ids:
        transition_key = ReferenceTransitionKey(record.asset_id, record.sequence_id, step_id)
        materialized = bridge.materialize(transition_key)
        sample = materialized.training_sample.sample_record
        if materialized.training_sample.trajectory_id != trajectory_id:
            raise ValueError("bridge trajectory identity differs from the sequence record")
        if sample.ordinal != step_id:
            raise ValueError("bridge sample ordinal differs from the sequence transition")
        if topology_sha256 is None:
            topology_sha256 = sample.topology_sha256
            operator_geometry_sha256 = sample.operator_geometry_sha256
            material_sha256 = sample.material_sha256
        elif (
            sample.topology_sha256 != topology_sha256
            or sample.operator_geometry_sha256 != operator_geometry_sha256
            or sample.material_sha256 != material_sha256
        ):
            raise ValueError("one sequence resolved to conflicting v5 static identities")
        lookup_key = (trajectory_id, sample.sample_id)
        if lookup_key in transition_bindings:
            raise ValueError("duplicate bridge sample identity")
        transition_bindings[lookup_key] = transition_key
        sample_records.append(sample)
        del materialized

    if topology_sha256 is None or operator_geometry_sha256 is None or material_sha256 is None:
        raise RuntimeError("authenticated sequence unexpectedly contained no transitions")
    load_program = _load_program_payload(provenance)
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        scene_family=f"reference-sequence:{record.asset_id}",
        load_program_id=trajectory_id,
        load_program_sha256=canonical_json_sha256(load_program),
        source_chain_sha256=canonical_json_sha256(_source_chain_payload(provenance)),
        topology_sha256=topology_sha256,
        operator_geometry_sha256=operator_geometry_sha256,
        material_sha256=material_sha256,
        provenance=provenance,
        source_transition_count=provenance.source_transition_count,
        samples=tuple(sample_records),
    )


def build_reference_sequence_v5_corpus(
    dataset: ReferenceSequenceDataset,
    bridge: ReferenceSequenceV5Bridge,
) -> ReferenceSequenceV5Corpus:
    """Build complete v5 metadata while retaining no dynamic training samples."""
    if type(dataset) is not ReferenceSequenceDataset:
        raise TypeError("dataset must be a canonical ReferenceSequenceDataset")
    if type(bridge) is not ReferenceSequenceV5Bridge:
        raise TypeError("bridge must be a canonical ReferenceSequenceV5Bridge")
    if bridge.dataset is not dataset:
        raise ValueError("bridge must own the exact dataset")

    transition_bindings: dict[tuple[str, str], ReferenceTransitionKey] = {}
    records_by_role: dict[DatasetRole, tuple[TrajectoryRecord, ...]] = {}
    for role in DatasetRole:
        records_by_role[role] = tuple(
            _build_trajectory(dataset, bridge, record, transition_bindings) for record in dataset.records(role)
        )
    manifest = SplitManifest(
        train=records_by_role[DatasetRole.TRAIN],
        validation=records_by_role[DatasetRole.VALIDATION],
        confirmation=records_by_role[DatasetRole.CONFIRMATION],
        consumed_regression=records_by_role[DatasetRole.CONSUMED_REGRESSION],
    )
    return ReferenceSequenceV5Corpus(
        split_manifest=manifest,
        transition_keys_by_sample=transition_bindings,
        source_index_sha256=dataset.index_sha256,
    )


__all__ = [
    "ReferenceSequenceSplitIndexBuild",
    "ReferenceSequenceV5Corpus",
    "build_reference_sequence_split_index",
    "build_reference_sequence_v5_corpus",
    "write_reference_sequence_split_index",
]
