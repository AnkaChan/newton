# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for producer and consumer portable reference-sequence metadata."""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .. import reference_sequence_portable_corpus as corpus_contract
from ..reference_sequence_dataset import ReferenceSequenceDataset, ReferenceTransitionKey
from ..reference_sequence_portable_corpus import (
    ReferenceSequencePortableConsumerView,
    ReferenceSequencePortableCorpus,
    materialize_reference_sequence_portable_corpus,
)
from ..reference_sequence_v5_bridge import ReferenceSequencePortableDatasetBridge
from ..v5_dataset import DatasetRole
from .test_reference_sequence_dataset import _write_index, _write_sequence_record


def _dataset(root: Path) -> ReferenceSequenceDataset:
    records = [
        _write_sequence_record(
            root,
            role=DatasetRole.TRAIN,
            asset_id="train-asset",
            sequence_id="sample-000",
            deformation_phase=0.11,
            deformation_scale=1.1,
        ),
        _write_sequence_record(
            root,
            role=DatasetRole.VALIDATION,
            asset_id="validation-asset",
            sequence_id="sample-000",
            offset=3.0,
            deformation_phase=0.29,
            deformation_scale=1.7,
        ),
        _write_sequence_record(
            root,
            role=DatasetRole.CONFIRMATION,
            asset_id="confirmation-asset",
            sequence_id="sample-000",
            offset=6.0,
            deformation_phase=0.47,
            deformation_scale=2.3,
        ),
    ]
    return ReferenceSequenceDataset.load(_write_index(root, records))


class TestReferenceSequencePortableCorpus(unittest.TestCase):
    def test_producer_materialization_records_roles_and_retains_no_dynamic_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            bridge = ReferenceSequencePortableDatasetBridge(dataset, device="cpu")

            corpus = materialize_reference_sequence_portable_corpus(dataset, bridge)

            self.assertIs(type(corpus), ReferenceSequencePortableCorpus)
            self.assertEqual(corpus.prepared_roles, tuple(DatasetRole))
            self.assertEqual(len(corpus.split_manifest.train), 1)
            self.assertEqual(len(corpus.split_manifest.validation), 1)
            self.assertEqual(len(corpus.split_manifest.confirmation), 1)
            self.assertEqual(len(corpus.transition_keys_by_sample), 9)
            self.assertNotIn("materialized_sample", vars(corpus))
            self.assertNotIn("bridge", vars(corpus))
            receipt = corpus.as_dict()["producer_preparation"]
            self.assertEqual(receipt["prepared_roles"], [role.value for role in DatasetRole])
            self.assertEqual(
                receipt["claim_scope"],
                "complete-requested-role-metadata-not-proof-of-payload-opens-or-access-control",
            )
            self.assertEqual(
                corpus.source_role_inventories[DatasetRole.CONSUMED_REGRESSION].source_transition_count,
                0,
            )

    def test_consumer_reconstructs_full_metadata_and_train_validation_view_without_confirmation_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            producer_dataset = _dataset(root)
            producer = materialize_reference_sequence_portable_corpus(
                producer_dataset,
                ReferenceSequencePortableDatasetBridge(producer_dataset, device="cpu"),
            )
            sealed_json = producer.to_json()
            trusted_corpus_sha256 = producer.corpus_sha256

            loaded_roles: list[DatasetRole] = []
            original_transition = ReferenceSequenceDataset.transition

            def tracked_transition(instance, key):
                transition = original_transition(instance, key)
                loaded_roles.append(transition.role)
                return transition

            with mock.patch.object(ReferenceSequenceDataset, "transition", new=tracked_transition):
                consumer_corpus = ReferenceSequencePortableCorpus.from_json(sealed_json)
                view = consumer_corpus.consumer_view()

            self.assertEqual(loaded_roles, [])
            self.assertIs(type(view), ReferenceSequencePortableConsumerView)
            self.assertEqual(view.roles, (DatasetRole.TRAIN, DatasetRole.VALIDATION))
            self.assertEqual(len(view.records(DatasetRole.TRAIN)), 1)
            self.assertEqual(len(view.records(DatasetRole.VALIDATION)), 1)
            self.assertEqual(len(view.transition_keys_by_sample), 6)
            self.assertEqual(len(consumer_corpus.split_manifest.confirmation), 1)
            self.assertNotIn(DatasetRole.CONFIRMATION, loaded_roles)
            self.assertEqual(
                ReferenceSequencePortableConsumerView.from_json(
                    view.to_json(),
                    authenticated_source_corpus=consumer_corpus,
                    trusted_source_corpus_sha256=trusted_corpus_sha256,
                ),
                view,
            )

            with self.assertRaisesRegex(ValueError, "consumer view may contain only train and validation"):
                consumer_corpus.consumer_view((DatasetRole.TRAIN, DatasetRole.CONFIRMATION))

    def test_consumer_on_demand_training_load_does_not_touch_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            producer = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
            )
            consumer = ReferenceSequencePortableCorpus.from_json(producer.to_json()).consumer_view()
            key = next(iter(consumer.transition_keys_by_sample.values()))

            loaded_roles: list[DatasetRole] = []
            original_transition = ReferenceSequenceDataset.transition

            def tracked_transition(instance, transition_key):
                transition = original_transition(instance, transition_key)
                loaded_roles.append(transition.role)
                return transition

            with mock.patch.object(ReferenceSequenceDataset, "transition", new=tracked_transition):
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu").materialize(key)
            self.assertEqual(loaded_roles, [DatasetRole.TRAIN])
            self.assertNotIn(DatasetRole.CONFIRMATION, loaded_roles)

    def test_consumer_view_rejects_transition_value_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
                roles=(DatasetRole.TRAIN,),
            )
            view = corpus.consumer_view((DatasetRole.TRAIN,))
            bindings = dict(view.transition_keys_by_sample)
            lookup = next(key for key in bindings if key[1] == "step-00000000")
            source = bindings[lookup]
            bindings[lookup] = ReferenceTransitionKey(source.asset_id, source.sequence_id, 1)

            with self.assertRaisesRegex(ValueError, "transition-key bindings"):
                dataclasses.replace(view, transition_keys_by_sample=bindings)

    def test_consumer_view_rejects_cross_role_asset_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
            )
            view = corpus.consumer_view()
            train = view.records(DatasetRole.TRAIN)
            train_lookups = {
                sample_lookup
                for sample_lookup in view.transition_keys_by_sample
                if sample_lookup[0] == train[0].trajectory_id
            }
            bindings = {
                lookup: key for lookup, key in view.transition_keys_by_sample.items() if lookup in train_lookups
            }

            with self.assertRaisesRegex(ValueError, "across roles"):
                dataclasses.replace(
                    view,
                    records_by_role={
                        DatasetRole.TRAIN: train,
                        DatasetRole.VALIDATION: train,
                    },
                    transition_keys_by_sample=bindings,
                )

    def test_consumer_view_requires_authenticated_exact_source_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
            )
            view = corpus.consumer_view((DatasetRole.TRAIN,))

            with self.assertRaisesRegex(ValueError, "externally authenticated root"):
                ReferenceSequencePortableConsumerView.from_json(
                    view.to_json(),
                    authenticated_source_corpus=corpus,
                    trusted_source_corpus_sha256="f" * 64,
                )

            validation_record = corpus.split_manifest.validation[0]
            validation_lookups = {
                (validation_record.trajectory_id, sample.sample_id) for sample in validation_record.samples
            }
            forged = ReferenceSequencePortableConsumerView(
                source_corpus_sha256=corpus.corpus_sha256,
                source_manifest_sha256=corpus.split_manifest.manifest_sha256,
                roles=(DatasetRole.TRAIN,),
                records_by_role={DatasetRole.TRAIN: (validation_record,)},
                transition_keys_by_sample={
                    lookup: corpus.transition_keys_by_sample[lookup] for lookup in validation_lookups
                },
            )
            with self.assertRaisesRegex(ValueError, "exact authenticated corpus projection"):
                forged.validate_authenticated_projection(
                    corpus,
                    trusted_source_corpus_sha256=corpus.corpus_sha256,
                )

    def test_role_filtered_preparation_never_opens_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            loaded_roles: list[DatasetRole] = []
            original_transition = ReferenceSequenceDataset.transition

            def tracked_transition(instance, transition_key):
                transition = original_transition(instance, transition_key)
                loaded_roles.append(transition.role)
                return transition

            with mock.patch.object(ReferenceSequenceDataset, "transition", new=tracked_transition):
                corpus = materialize_reference_sequence_portable_corpus(
                    dataset,
                    ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
                    roles=(DatasetRole.TRAIN, DatasetRole.VALIDATION),
                )

            self.assertEqual(corpus.prepared_roles, (DatasetRole.TRAIN, DatasetRole.VALIDATION))
            self.assertEqual(set(loaded_roles), {DatasetRole.TRAIN, DatasetRole.VALIDATION})
            self.assertNotIn(DatasetRole.CONFIRMATION, loaded_roles)
            self.assertEqual(corpus.split_manifest.confirmation, ())
            self.assertEqual(corpus.consumer_view().roles, (DatasetRole.TRAIN, DatasetRole.VALIDATION))

    def test_preparation_preserves_legitimate_requested_empty_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
                roles=(DatasetRole.TRAIN, DatasetRole.CONSUMED_REGRESSION),
            )

            self.assertEqual(corpus.prepared_roles, (DatasetRole.TRAIN, DatasetRole.CONSUMED_REGRESSION))
            self.assertEqual(corpus.split_manifest.consumed_regression, ())
            self.assertEqual(
                corpus.source_role_inventories[DatasetRole.CONSUMED_REGRESSION].trajectory_count,
                0,
            )

    def test_receipt_cannot_overstate_nonempty_unmaterialized_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
                roles=(DatasetRole.TRAIN,),
            )

            with self.assertRaisesRegex(ValueError, "materialization"):
                dataclasses.replace(
                    corpus,
                    prepared_roles=(DatasetRole.TRAIN, DatasetRole.VALIDATION, DatasetRole.CONFIRMATION),
                )

    def test_strict_corpus_deserialization_rejects_extra_duplicate_and_tampered_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
                roles=(DatasetRole.TRAIN,),
            )
            self.assertEqual(corpus.prepared_roles, (DatasetRole.TRAIN,))
            payload = corpus.as_dict()
            payload["extra"] = True
            with self.assertRaisesRegex(ValueError, "keys must be exactly"):
                ReferenceSequencePortableCorpus.from_dict(payload)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                ReferenceSequencePortableCorpus.from_json('{"schema_version":1,"schema_version":1}')
            tampered = json.loads(corpus.to_json())
            tampered["transition_keys"][0]["step_id"] += 1
            with self.assertRaisesRegex(ValueError, "transition-key bindings"):
                ReferenceSequencePortableCorpus.from_dict(tampered)

            inventories = dict(corpus.source_role_inventories)
            train_inventory = inventories[DatasetRole.TRAIN]
            changed_sequence = dataclasses.replace(
                train_inventory.sequences[0],
                producer_manifest_json="different-manifest.json",
            )
            inventories[DatasetRole.TRAIN] = dataclasses.replace(
                train_inventory,
                sequences=(changed_sequence,),
            )
            with self.assertRaisesRegex(ValueError, "authenticated source index"):
                dataclasses.replace(corpus, source_role_inventories=inventories)

            view_payload = corpus.consumer_view((DatasetRole.TRAIN,)).as_dict()
            view_payload["records"]["train"][0]["operator_volume"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "portable static identity"):
                ReferenceSequencePortableConsumerView.from_dict(
                    view_payload,
                    authenticated_source_corpus=corpus,
                    trusted_source_corpus_sha256=corpus.corpus_sha256,
                )

    def test_corpus_identity_is_filesystem_relocation_stable(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_root = Path(first_directory)
            first_dataset = _dataset(first_root)
            first = materialize_reference_sequence_portable_corpus(
                first_dataset,
                ReferenceSequencePortableDatasetBridge(first_dataset, device="cpu"),
                roles=(DatasetRole.TRAIN,),
            )
            second_root = Path(second_directory) / "relocated"
            shutil.copytree(first_root, second_root)
            second_dataset = ReferenceSequenceDataset.load(second_root / "index.json")
            second = materialize_reference_sequence_portable_corpus(
                second_dataset,
                ReferenceSequencePortableDatasetBridge(second_dataset, device="cpu"),
                roles=(DatasetRole.TRAIN,),
            )
            self.assertNotEqual(first_dataset.index_path, second_dataset.index_path)
            self.assertEqual(first.as_dict(), second.as_dict())
            self.assertEqual(first.corpus_sha256, second.corpus_sha256)

    def test_corpus_rejects_reference_sequence_provenance_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
                roles=(DatasetRole.TRAIN,),
            )
            relabelled = dataclasses.replace(
                corpus.split_manifest.train[0],
                scene_family="unbound-scene",
            )
            relabelled_manifest = dataclasses.replace(
                corpus.split_manifest,
                train=(relabelled,),
            )
            with self.assertRaisesRegex(ValueError, "reference-sequence provenance"):
                dataclasses.replace(corpus, split_manifest=relabelled_manifest)

    def test_producer_rejects_materialized_source_identity_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            bridge = ReferenceSequencePortableDatasetBridge(dataset, device="cpu")
            original_materialize = bridge.materialize

            def tampered_materialize(key):
                materialized = original_materialize(key)
                object.__setattr__(
                    materialized,
                    "identities",
                    dataclasses.replace(materialized.identities, static_npz_sha256="f" * 64),
                )
                return materialized

            with (
                mock.patch.object(bridge, "materialize", side_effect=tampered_materialize),
                self.assertRaisesRegex(ValueError, "materialized.*source"),
            ):
                materialize_reference_sequence_portable_corpus(
                    dataset,
                    bridge,
                    roles=(DatasetRole.TRAIN,),
                )

    def test_corpus_rejects_coherent_source_provenance_relabels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            corpus = materialize_reference_sequence_portable_corpus(
                dataset,
                ReferenceSequencePortableDatasetBridge(dataset, device="cpu"),
                roles=(DatasetRole.TRAIN,),
            )
            trajectory = corpus.split_manifest.train[0]
            cases = {
                "asset source": {"asset_source_sha256": "f" * 64},
                "static bundle": {"static_bundle_sha256": "e" * 64},
                "producer topology": {"producer_topology_sha256": "d" * 64},
                "accepted reference": {
                    "accepted_reference_state_sha256": (
                        "c" * 64,
                        *trajectory.provenance.accepted_reference_state_sha256[1:],
                    )
                },
            }
            for label, changes in cases.items():
                relabelled_provenance = dataclasses.replace(trajectory.provenance, **changes)
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, "source transition"):
                    relabelled_trajectory = dataclasses.replace(
                        trajectory,
                        provenance=relabelled_provenance,
                        load_program_sha256=corpus_contract.canonical_json_sha256(
                            corpus_contract._load_program_payload(relabelled_provenance)
                        ),
                        source_chain_sha256=corpus_contract.canonical_json_sha256(
                            corpus_contract._source_chain_payload(relabelled_provenance)
                        ),
                    )
                    relabelled_manifest = dataclasses.replace(
                        corpus.split_manifest,
                        train=(relabelled_trajectory,),
                    )
                    dataclasses.replace(corpus, split_manifest=relabelled_manifest)


if __name__ == "__main__":
    unittest.main()
