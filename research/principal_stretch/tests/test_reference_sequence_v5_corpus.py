# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for sequence-native v5 split-manifest construction."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..reference_sequence_dataset import ReferenceSequenceDataset, ReferenceTransitionKey
from ..reference_sequence_v5_bridge import ReferenceSequenceV5Bridge
from ..reference_sequence_v5_corpus import (
    ReferenceSequenceSplitIndexBuild,
    ReferenceSequenceV5Corpus,
    build_reference_sequence_split_index,
    build_reference_sequence_v5_corpus,
    write_reference_sequence_split_index,
)
from ..v5_dataset import DataAccessLedger, DatasetRole, ReferenceSequenceProvenance, build_sampling_schedule
from .test_reference_sequence_dataset import _rewrite_sequence_npz, _write_index, _write_sequence_record

_ROLE_ASSETS = {
    DatasetRole.TRAIN: ("cube", "torus"),
    DatasetRole.VALIDATION: ("bunny_small",),
    DatasetRole.CONFIRMATION: ("ditto", "thin_sheet"),
}


def _five_by_three_records(root: Path) -> list[dict[str, object]]:
    records = []
    for role_index, (role, assets) in enumerate(_ROLE_ASSETS.items(), start=1):
        for asset_index, asset_id in enumerate(assets, start=1):
            for sequence_index in range(3):
                records.append(
                    _write_sequence_record(
                        root,
                        role=role,
                        asset_id=asset_id,
                        sequence_id=f"sample-{sequence_index:03d}",
                        offset=float(10 * role_index + asset_index),
                        deformation_phase=0.13 * role_index + 0.017 * asset_index,
                        deformation_scale=1.0 + 0.31 * role_index + 0.07 * asset_index,
                    )
                )
    return records


def _nested_producer_index(root: Path, records: list[dict[str, object]], *, reverse: bool = False) -> Path:
    records_by_asset: dict[str, list[dict[str, object]]] = {}
    for record in records:
        records_by_asset.setdefault(str(record["asset_id"]), []).append(record)
    assets = []
    protocol = None
    for asset_id in sorted(records_by_asset, reverse=reverse):
        sequences = []
        asset_manifest = None
        for record in sorted(
            records_by_asset[asset_id],
            key=lambda value: str(value["sequence_id"]),
            reverse=reverse,
        ):
            manifest_path = root / str(record["producer_manifest_json"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            asset_manifest = manifest
            protocol = manifest["protocol"]
            sequence_path = manifest_path.parent / manifest["files"]["sequence_npz"]["path"]
            sequences.append(
                {
                    "sequence_id": record["sequence_id"],
                    "deformation_seed": manifest["deformation_seed"],
                    "velocity_seed": manifest["velocity_seed"],
                    "manifest": {
                        "path": manifest_path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    },
                    "sequence_npz": {
                        "path": sequence_path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
                    },
                }
            )
        assert asset_manifest is not None
        manifest_path = root / str(records_by_asset[asset_id][0]["producer_manifest_json"])
        static_path = manifest_path.parent / asset_manifest["files"]["static_npz"]["path"]
        with np.load(static_path, allow_pickle=False) as archive:
            vertex_count = int(archive["rest_q"].shape[0])
            tet_count = int(archive["tet_indices"].shape[0])
        assets.append(
            {
                "asset_id": asset_id,
                "source": asset_manifest["source"],
                "source_sha256": asset_manifest["source_sha256"],
                "vertex_count": vertex_count,
                "tet_count": tet_count,
                "static_npz": {
                    "path": static_path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
                },
                "identities": asset_manifest["identities"],
                "sequences": sequences,
            }
        )
    assert protocol is not None
    payload = {
        "schema": "pss-free-body-reference-index-v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "base_seed": 2026081601,
        "samples_per_asset": len(next(iter(records_by_asset.values()))),
        "hierarchy_config": {"n_levels": 2, "cluster_size": 8},
        "asset_count": len(assets),
        "accepted_sequence_count": sum(len(asset["sequences"]) for asset in assets),
        "assets": assets,
    }
    path = root / "index.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


class TestReferenceSequenceV5Corpus(unittest.TestCase):
    def test_split_index_build_itself_runs_the_strict_flat_loader(self):
        for tamper in ("manifest-inventory", "rejected-evidence"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                record = _write_sequence_record(
                    root,
                    role=DatasetRole.TRAIN,
                    asset_id="cube",
                    sequence_id="sample-000",
                )
                producer_index = _nested_producer_index(root, [record])
                nested = json.loads(producer_index.read_text(encoding="utf-8"))
                nested_sequence = nested["assets"][0]["sequences"][0]
                manifest_path = root / nested_sequence["manifest"]["path"]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_error = "producer shard inventory_sha256 mismatch"
                if tamper == "manifest-inventory":
                    manifest["inventory_sha256"] = "0" * 64
                else:
                    evidence_record = manifest["files"]["evidence_json"]
                    evidence_path = manifest_path.parent / evidence_record["path"]
                    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    evidence["steps"][0]["reference_accepted"] = False
                    evidence_bytes = (
                        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    ).encode("utf-8")
                    evidence_path.write_bytes(evidence_bytes)
                    evidence_record["bytes"] = len(evidence_bytes)
                    evidence_record["sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
                    manifest["inventory_sha256"] = hashlib.sha256(
                        json.dumps(
                            {"files": manifest["files"], "identities": manifest["identities"]},
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    expected_error = "training split must not contain rejected reference evidence"
                manifest_bytes = (
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode("utf-8")
                manifest_path.write_bytes(manifest_bytes)
                nested_sequence["manifest"]["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
                producer_index.write_text(json.dumps(nested, allow_nan=False), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, expected_error):
                    build_reference_sequence_split_index(
                        producer_index,
                        asset_roles={"cube": DatasetRole.TRAIN},
                    )
                self.assertEqual(tuple(root.glob(".reference-sequence-split-index.json.build-validation.*.tmp")), ())

    def test_nested_producer_index_builds_deterministic_flat_split_index_beside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _five_by_three_records(root)
            producer_index = _nested_producer_index(root, list(reversed(records)), reverse=True)
            asset_roles = {asset_id: role for role, assets in _ROLE_ASSETS.items() for asset_id in assets}

            built = build_reference_sequence_split_index(producer_index, asset_roles=asset_roles)
            self.assertIs(type(built), ReferenceSequenceSplitIndexBuild)
            self.assertEqual(built.producer_index_path, producer_index.resolve())
            self.assertEqual(built.split_index_path, root / "reference-sequence-split-index.json")
            self.assertFalse(built.split_index_path.exists())
            self.assertEqual(built.asset_count, 5)
            self.assertEqual(built.sequence_count, 15)
            self.assertEqual(built.split_index_file_sha256, hashlib.sha256(built.split_index_bytes).hexdigest())

            written = write_reference_sequence_split_index(producer_index, asset_roles=asset_roles)
            self.assertEqual(written, built)
            self.assertEqual(written.split_index_path.read_bytes(), written.split_index_bytes)
            dataset = ReferenceSequenceDataset.load(written.split_index_path)
            self.assertEqual(dataset.index_sha256, written.dataset_index_sha256)
            self.assertEqual(len(dataset.records(DatasetRole.TRAIN)), 6)
            self.assertEqual(len(dataset.records(DatasetRole.VALIDATION)), 3)
            self.assertEqual(len(dataset.records(DatasetRole.CONFIRMATION)), 6)
            for role in DatasetRole:
                for record in dataset.records(role):
                    self.assertNotIn("..", Path(record.producer_manifest_json).parts)

            reordered_index = _nested_producer_index(root, records, reverse=False)
            reordered = build_reference_sequence_split_index(reordered_index, asset_roles=asset_roles)
            self.assertEqual(reordered.split_index_bytes, built.split_index_bytes)
            self.assertEqual(reordered.dataset_index_sha256, built.dataset_index_sha256)
            self.assertEqual(write_reference_sequence_split_index(reordered_index, asset_roles=asset_roles), reordered)

    def test_split_index_builder_rejects_role_map_hash_payload_and_output_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="cube",
                sequence_id="sample-000",
            )
            producer_index = _nested_producer_index(root, [record])
            with self.assertRaisesRegex(ValueError, "asset_roles must exactly cover producer assets"):
                build_reference_sequence_split_index(producer_index, asset_roles={})
            with self.assertRaisesRegex(ValueError, "asset_roles must exactly cover producer assets"):
                build_reference_sequence_split_index(
                    producer_index,
                    asset_roles={"cube": DatasetRole.TRAIN, "invented": DatasetRole.VALIDATION},
                )

            payload = json.loads(producer_index.read_text(encoding="utf-8"))
            payload["assets"][0]["sequences"][0]["sequence_npz"]["sha256"] = "0" * 64
            producer_index.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sequence_npz SHA-256 differs"):
                build_reference_sequence_split_index(producer_index, asset_roles={"cube": DatasetRole.TRAIN})

            producer_index = _nested_producer_index(root, [record])
            sequence_path = (
                root / json.loads(producer_index.read_text())["assets"][0]["sequences"][0]["sequence_npz"]["path"]
            )
            sequence_path.write_bytes(sequence_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "byte count differs|artifact SHA-256 mismatch"):
                build_reference_sequence_split_index(producer_index, asset_roles={"cube": DatasetRole.TRAIN})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="cube",
                sequence_id="sample-000",
            )
            _rewrite_sequence_npz(
                root,
                record,
                lambda arrays: arrays["q"].__setitem__((1, 0, 0), np.nan),
            )
            producer_index = _nested_producer_index(root, [record])
            with self.assertRaisesRegex(ValueError, "reference q must contain only finite values"):
                build_reference_sequence_split_index(producer_index, asset_roles={"cube": DatasetRole.TRAIN})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="cube",
                sequence_id="sample-000",
            )
            producer_index = _nested_producer_index(root, [record])
            output = root / "reference-sequence-split-index.json"
            output.write_bytes(b"pre-existing-different-bytes")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite non-identical split index"):
                write_reference_sequence_split_index(producer_index, asset_roles={"cube": DatasetRole.TRAIN})

    def test_builds_one_complete_trajectory_per_sequence_without_retaining_training_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _five_by_three_records(root)
            dataset = ReferenceSequenceDataset.load(_write_index(root, list(reversed(records))))
            bridge = ReferenceSequenceV5Bridge(dataset, device="cpu")
            corpus = build_reference_sequence_v5_corpus(dataset, bridge)

            self.assertIs(type(corpus), ReferenceSequenceV5Corpus)
            self.assertEqual(corpus.source_index_sha256, dataset.index_sha256)
            self.assertEqual(len(corpus.split_manifest.train), 6)
            self.assertEqual(len(corpus.split_manifest.validation), 3)
            self.assertEqual(len(corpus.split_manifest.confirmation), 6)
            self.assertEqual(len(corpus.transition_keys_by_sample), 45)
            self.assertEqual(
                {trajectory.scene_family for trajectory in corpus.split_manifest.train},
                {"reference-sequence:cube", "reference-sequence:torus"},
            )
            for role in (DatasetRole.TRAIN, DatasetRole.VALIDATION, DatasetRole.CONFIRMATION):
                for trajectory in corpus.split_manifest.records(role):
                    self.assertIs(type(trajectory.provenance), ReferenceSequenceProvenance)
                    self.assertEqual(trajectory.source_transition_count, 3)
                    self.assertEqual(tuple(sample.ordinal for sample in trajectory.samples), (0, 1, 2))
                    for sample in trajectory.samples:
                        key = corpus.transition_keys_by_sample[(trajectory.trajectory_id, sample.sample_id)]
                        self.assertEqual(
                            key,
                            ReferenceTransitionKey(
                                asset_id=trajectory.provenance.asset_id,
                                sequence_id=trajectory.provenance.sequence_id,
                                step_id=sample.ordinal,
                            ),
                        )
            self.assertNotIn("training_sample", vars(corpus))
            with self.assertRaises(TypeError):
                corpus.transition_keys_by_sample[("forged", "step-00000000")] = ReferenceTransitionKey(
                    "cube", "sample-000", 0
                )

            schedule = build_sampling_schedule(
                corpus.split_manifest,
                role=DatasetRole.TRAIN,
                steps=6,
                batch_size=1,
                seed=2026081701,
            )
            self.assertEqual(schedule.trajectory_count, 6)
            self.assertEqual(DataAccessLedger(corpus.split_manifest).accesses, ())

            provenance = corpus.split_manifest.train[0].provenance
            original_evidence_sha256 = provenance.evidence_sha256
            object.__setattr__(provenance, "evidence_sha256", hashlib.sha256(b"tampered-evidence").hexdigest())
            try:
                with self.assertRaisesRegex(ValueError, "split manifest changed after authentication"):
                    ReferenceSequenceV5Corpus(
                        split_manifest=corpus.split_manifest,
                        transition_keys_by_sample=corpus.transition_keys_by_sample,
                        source_index_sha256=corpus.source_index_sha256,
                    )
            finally:
                object.__setattr__(provenance, "evidence_sha256", original_evidence_sha256)

            original_manifest_sha256 = corpus.split_manifest.manifest_sha256
            object.__setattr__(corpus.split_manifest, "manifest_sha256", "0" * 64)
            try:
                with self.assertRaisesRegex(ValueError, "split manifest changed after authentication"):
                    ReferenceSequenceV5Corpus(
                        split_manifest=corpus.split_manifest,
                        transition_keys_by_sample=corpus.transition_keys_by_sample,
                        source_index_sha256=corpus.source_index_sha256,
                    )
            finally:
                object.__setattr__(corpus.split_manifest, "manifest_sha256", original_manifest_sha256)

    def test_rejects_bridge_for_a_different_authenticated_dataset(self):
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_root = Path(first_directory)
            second_root = Path(second_directory)
            first = ReferenceSequenceDataset.load(
                _write_index(
                    first_root,
                    [
                        _write_sequence_record(
                            first_root,
                            role=DatasetRole.TRAIN,
                            asset_id="first",
                            sequence_id="sample-000",
                        )
                    ],
                )
            )
            second = ReferenceSequenceDataset.load(
                _write_index(
                    second_root,
                    [
                        _write_sequence_record(
                            second_root,
                            role=DatasetRole.TRAIN,
                            asset_id="second",
                            sequence_id="sample-000",
                        )
                    ],
                )
            )
            with self.assertRaisesRegex(ValueError, "bridge must own the exact dataset"):
                build_reference_sequence_v5_corpus(first, ReferenceSequenceV5Bridge(second, device="cpu"))


if __name__ == "__main__":
    unittest.main()
