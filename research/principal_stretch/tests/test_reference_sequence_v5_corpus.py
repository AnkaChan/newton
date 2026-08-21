# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for sequence-native v5 split-manifest construction."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ..reference_sequence_dataset import ReferenceSequenceDataset, ReferenceTransitionKey
from ..reference_sequence_v5_bridge import ReferenceSequenceV5Bridge
from ..reference_sequence_v5_corpus import ReferenceSequenceV5Corpus, build_reference_sequence_v5_corpus
from ..v5_dataset import DataAccessLedger, DatasetRole, ReferenceSequenceProvenance, build_sampling_schedule
from .test_reference_sequence_dataset import _write_index, _write_sequence_record


class TestReferenceSequenceV5Corpus(unittest.TestCase):
    def test_builds_one_complete_trajectory_per_sequence_without_retaining_training_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            role_assets = {
                DatasetRole.TRAIN: ("cube", "torus"),
                DatasetRole.VALIDATION: ("bunny_small",),
                DatasetRole.CONFIRMATION: ("ditto", "thin_sheet"),
            }
            records = []
            for role_index, (role, assets) in enumerate(role_assets.items(), start=1):
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
