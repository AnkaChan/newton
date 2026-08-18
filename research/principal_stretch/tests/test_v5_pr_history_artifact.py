# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import tempfile
import unittest
import unittest.mock
import zipfile

import numpy as np

from ..iterative_solver import PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32
from ..pr_scene_history import AtomicCoordinate, PRSceneHistory, _array_digest
from ..v5_checkpoint import canonical_json_sha256
from ..v5_pr_history_artifact import (
    _npy_bytes,
    _safe_npy_array,
    load_pr_history_v5_artifact,
    write_pr_history_v5_artifact,
)


def _canonical_source_bytes(source):
    return json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


class TestV5PRHistoryArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = PRSceneHistory("stretch")
        cls.chain = cls.history.generate(stop=AtomicCoordinate.from_ordinal(2), max_transitions=2)

    def _write(self, root: pathlib.Path, stem: str):
        return write_pr_history_v5_artifact(
            self.history,
            self.chain,
            selected_start_ordinal=0,
            selected_stop_ordinal=2,
            trajectory_id="unittest-pr-stretch-two",
            bundle_path=root / f"{stem}.npz",
            source_path=root / f"{stem}.json",
            bundle_uri="artifact://unittest/pr-stretch-two.npz",
            source_uri="source://unittest/pr-stretch-two.json",
            expected_history_chain_sha256=self.chain.chain_sha256,
            expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
            max_chain_transitions=2,
        )

    def _load(self, artifact, **kwargs):
        return load_pr_history_v5_artifact(
            artifact.source_path,
            artifact.bundle_path,
            expected_source_file_sha256=artifact.source_file_sha256,
            expected_bundle_file_sha256=artifact.bundle_file_sha256,
            expected_history_chain_sha256=self.chain.chain_sha256,
            expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
            max_chain_transitions=2,
            **kwargs,
        )

    @staticmethod
    def _write_forged_source(root, stem, source):
        source_payload = dict(source)
        source_payload.pop("source_record_sha256", None)
        source["source_record_sha256"] = canonical_json_sha256(source_payload)
        source_bytes = _canonical_source_bytes(source)
        source_path = root / f"{stem}.json"
        source_path.write_bytes(source_bytes)
        return source_path, hashlib.sha256(source_bytes).hexdigest()

    def _retarget_bundle(self, root, artifact, stem, bundle_bytes):
        bundle_path = root / f"{stem}.npz"
        bundle_path.write_bytes(bundle_bytes)
        bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        source = json.loads(artifact.source_path.read_bytes())
        source["bundle_sha256"] = bundle_sha256
        source_path, source_sha256 = self._write_forged_source(root, stem, source)
        return source_path, source_sha256, bundle_path, bundle_sha256

    def _load_paths(self, source_path, source_sha256, bundle_path, bundle_sha256, **kwargs):
        return load_pr_history_v5_artifact(
            source_path,
            bundle_path,
            expected_source_file_sha256=source_sha256,
            expected_bundle_file_sha256=bundle_sha256,
            expected_history_chain_sha256=self.chain.chain_sha256,
            expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
            max_chain_transitions=2,
            **kwargs,
        )

    @staticmethod
    def _write_canonical_zip(path, entries):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for entry_name, entry_raw in sorted(entries):
                info = zipfile.ZipInfo(entry_name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, entry_raw)

    def test_two_writes_are_byte_deterministic_and_round_trip(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            first = self._write(root, "first")
            second = self._write(root, "second")
            self.assertEqual(first.bundle_path.read_bytes(), second.bundle_path.read_bytes())
            self.assertEqual(first.source_path.read_bytes(), second.source_path.read_bytes())
            self.assertEqual(first.bundle_file_sha256, second.bundle_file_sha256)
            self.assertEqual(first.source_file_sha256, second.source_file_sha256)
            self.assertEqual(first.trajectory.as_dict(), second.trajectory.as_dict())

            loaded = self._load(first)
            loaded.validate_immutable()
            self.assertEqual(loaded.source_chain_sha256, self.chain.chain_sha256)
            self.assertFalse(hasattr(loaded, "chain"))
            self.assertFalse(hasattr(loaded, "history"))
            self.assertEqual(loaded.trajectory.as_dict(), first.trajectory.as_dict())
            self.assertEqual(len(loaded.loaded_samples), 2)
            samples = tuple(item.training_sample.sample_record for item in loaded.loaded_samples)
            self.assertEqual(tuple(sample.ordinal for sample in samples), (0, 1))
            self.assertTrue(
                all(sample.operator_geometry_sha256 == loaded.trajectory.operator_geometry_sha256 for sample in samples)
            )
            self.assertEqual(loaded.trajectory.source_transition_count, 2)
            self.assertEqual(loaded.trajectory.selection_contract, "complete-contiguous-trajectory-v1")
            self.assertTrue(loaded.current_code_compatibility.compatible)

            with unittest.mock.patch(
                "research.principal_stretch.v5_pr_history_artifact.PRSceneHistory",
                side_effect=RuntimeError("simulated future checkout"),
            ):
                archival = self._load(first)
            self.assertFalse(archival.current_code_compatibility.compatible)
            self.assertIn("simulated future checkout", archival.current_code_compatibility.reason)
            self.assertEqual(archival.trajectory.as_dict(), first.trajectory.as_dict())

            class ExplodingCurrentHistory:
                @property
                def manifest(self):
                    raise RuntimeError("future manifest access changed")

            with unittest.mock.patch(
                "research.principal_stretch.v5_pr_history_artifact.PRSceneHistory",
                return_value=ExplodingCurrentHistory(),
            ):
                post_constructor_failure = self._load(first)
            self.assertFalse(post_constructor_failure.current_code_compatibility.compatible)
            self.assertIn("future manifest access changed", post_constructor_failure.current_code_compatibility.reason)

    def test_compression_root_ten_round_trip_and_rejects_integration_record_tamper(self):
        history = PRSceneHistory("compression-50")
        chain = history.generate(stop=AtomicCoordinate.from_ordinal(10), max_transitions=10)
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = write_pr_history_v5_artifact(
                history,
                chain,
                selected_start_ordinal=0,
                selected_stop_ordinal=10,
                trajectory_id="unittest-pr-compression-root-ten",
                bundle_path=root / "compression-root-ten.npz",
                source_path=root / "compression-root-ten.json",
                bundle_uri="artifact://unittest/pr-compression-root-ten.npz",
                source_uri="source://unittest/pr-compression-root-ten.json",
                expected_history_chain_sha256=chain.chain_sha256,
                expected_root_checkpoint_sha256=history.initial_checkpoint.checkpoint_sha256,
                max_chain_transitions=10,
            )
            loaded = load_pr_history_v5_artifact(
                artifact.source_path,
                artifact.bundle_path,
                expected_source_file_sha256=artifact.source_file_sha256,
                expected_bundle_file_sha256=artifact.bundle_file_sha256,
                expected_history_chain_sha256=chain.chain_sha256,
                expected_root_checkpoint_sha256=history.initial_checkpoint.checkpoint_sha256,
                max_chain_transitions=10,
            )
            self.assertEqual(len(loaded.loaded_samples), 10)
            ordinal_six = loaded.loaded_samples[6]
            self.assertEqual(ordinal_six.training_sample.sample_record.ordinal, 6)
            self.assertEqual(
                ordinal_six.physical_integration.integration_policy,
                PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            )
            self.assertEqual(
                ordinal_six.physical_integration.source_transition_sha256,
                chain.transitions[6].transition_sha256,
            )
            self.assertEqual(
                ordinal_six.physical_integration.source_evidence_sha256,
                ordinal_six.training_sample.physical_step.source_evidence.evidence_sha256,
            )
            loaded.validate_immutable()

            def policy_tamper(record):
                record["integration_policy"] = "position-history-execution-dtype-v1"

            def evidence_tamper(record):
                record["source_evidence_sha256"] = "0" * 64

            def transition_tamper(record):
                record["source_transition_sha256"] = chain.transitions[1].transition_sha256

            for variant, mutate in (
                ("integration-policy", policy_tamper),
                ("integration-evidence", evidence_tamper),
                ("integration-transition", transition_tamper),
            ):
                with self.subTest(variant=variant):
                    source = json.loads(artifact.source_path.read_bytes())
                    integration = source["samples"][0]["physical_integration"]
                    mutate(integration)
                    integration_payload = dict(integration)
                    integration_payload.pop("binding_sha256")
                    integration["binding_sha256"] = canonical_json_sha256(integration_payload)
                    selection = source["selection"]
                    selection["selected_physical_integration_binding_sha256"][0] = integration["binding_sha256"]
                    selection_payload = dict(selection)
                    selection_payload.pop("selection_sha256")
                    selection["selection_sha256"] = canonical_json_sha256(selection_payload)
                    source_path, source_sha256 = self._write_forged_source(root, variant, source)
                    with self.assertRaisesRegex(ValueError, "physical integration differs"):
                        load_pr_history_v5_artifact(
                            source_path,
                            artifact.bundle_path,
                            expected_source_file_sha256=source_sha256,
                            expected_bundle_file_sha256=artifact.bundle_file_sha256,
                            expected_history_chain_sha256=chain.chain_sha256,
                            expected_root_checkpoint_sha256=history.initial_checkpoint.checkpoint_sha256,
                            max_chain_transitions=10,
                        )

    def test_refuses_overwrite_external_hash_tamper_and_bounds(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")
            with self.assertRaises(FileExistsError):
                self._write(root, "artifact")
            with self.assertRaisesRegex(ValueError, "complete selection"):
                write_pr_history_v5_artifact(
                    self.history,
                    self.chain,
                    selected_start_ordinal=0,
                    selected_stop_ordinal=1,
                    trajectory_id="unittest-incomplete-selection",
                    bundle_path=root / "incomplete.npz",
                    source_path=root / "incomplete.json",
                    bundle_uri="artifact://unittest/incomplete.npz",
                    source_uri="source://unittest/incomplete.json",
                    expected_history_chain_sha256=self.chain.chain_sha256,
                    expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
                    max_chain_transitions=2,
                )
            with self.assertRaisesRegex(ValueError, "complete selection"):
                write_pr_history_v5_artifact(
                    self.history,
                    self.chain,
                    selected_start_ordinal=False,
                    selected_stop_ordinal=2,
                    trajectory_id="unittest-boolean-selection",
                    bundle_path=root / "boolean.npz",
                    source_path=root / "boolean.json",
                    bundle_uri="artifact://unittest/boolean.npz",
                    source_uri="source://unittest/boolean.json",
                    expected_history_chain_sha256=self.chain.chain_sha256,
                    expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
                    max_chain_transitions=2,
                )
            with self.assertRaisesRegex(ValueError, "source file differs"):
                load_pr_history_v5_artifact(
                    artifact.source_path,
                    artifact.bundle_path,
                    expected_source_file_sha256="0" * 64,
                    expected_bundle_file_sha256=artifact.bundle_file_sha256,
                    expected_history_chain_sha256=self.chain.chain_sha256,
                    expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
                    max_chain_transitions=2,
                )
            with self.assertRaisesRegex(ValueError, "source JSON bundle SHA-256"):
                load_pr_history_v5_artifact(
                    artifact.source_path,
                    artifact.bundle_path,
                    expected_source_file_sha256=artifact.source_file_sha256,
                    expected_bundle_file_sha256="0" * 64,
                    expected_history_chain_sha256=self.chain.chain_sha256,
                    expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
                    max_chain_transitions=2,
                )
            with self.assertRaisesRegex(ValueError, "max_entries"):
                self._load(artifact, max_entries=1)
            with self.assertRaisesRegex(ValueError, "max_source_bytes"):
                self._load(artifact, max_source_bytes=1)
            source = json.loads(artifact.source_path.read_bytes())
            selection = source["selection"]
            selection["selected_stop_ordinal_exclusive"] = 1
            selection["excluded_suffix_count"] = 1
            for name in (
                "selected_transition_sha256",
                "selected_sample_sha256",
                "selected_reference_acceptance_sha256",
                "selected_physical_integration_binding_sha256",
                "selected_operator_geometry_sha256",
                "selected_source_tet_poses_sha256",
            ):
                selection[name] = selection[name][:1]
            selection_payload = dict(selection)
            selection_payload.pop("selection_sha256")
            selection["selection_sha256"] = canonical_json_sha256(selection_payload)
            source_path, source_sha256 = self._write_forged_source(root, "incomplete-reader", source)
            with self.assertRaisesRegex(ValueError, "select the complete chain"):
                self._load_paths(
                    source_path,
                    source_sha256,
                    artifact.bundle_path,
                    artifact.bundle_file_sha256,
                )
            source = json.loads(artifact.source_path.read_bytes())
            source["selection"]["selected_start_ordinal"] = False
            selection_payload = dict(source["selection"])
            selection_payload.pop("selection_sha256")
            source["selection"]["selection_sha256"] = canonical_json_sha256(selection_payload)
            source_path, source_sha256 = self._write_forged_source(root, "boolean-reader", source)
            with self.assertRaisesRegex(ValueError, "select the complete chain"):
                self._load_paths(
                    source_path,
                    source_sha256,
                    artifact.bundle_path,
                    artifact.bundle_file_sha256,
                )

    def test_rejects_noncanonical_zip_order_metadata_and_npy_encoding(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")
            with zipfile.ZipFile(artifact.bundle_path, "r") as archive:
                entries = [(info.filename, archive.read(info)) for info in archive.infolist()]

            variants = {"order": list(reversed(entries)), "timestamp": entries}
            npy_entries = list(entries)
            name, raw = npy_entries[0]
            array = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            npy_entries[0] = (name, stream.getvalue())
            variants["npy"] = npy_entries

            for variant, variant_entries in variants.items():
                with self.subTest(variant=variant):
                    bundle = root / f"{variant}.npz"
                    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                        for index, (entry_name, entry_raw) in enumerate(variant_entries):
                            timestamp = (
                                (1981, 1, 1, 0, 0, 0)
                                if variant == "timestamp" and index == 0
                                else (1980, 1, 1, 0, 0, 0)
                            )
                            info = zipfile.ZipInfo(entry_name, date_time=timestamp)
                            info.compress_type = zipfile.ZIP_STORED
                            info.create_system = 3
                            info.external_attr = 0o600 << 16
                            archive.writestr(info, entry_raw)
                    bundle_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
                    source = json.loads(artifact.source_path.read_bytes())
                    source["bundle_sha256"] = bundle_sha256
                    source_payload = dict(source)
                    source_payload.pop("source_record_sha256")
                    source["source_record_sha256"] = canonical_json_sha256(source_payload)
                    source_bytes = _canonical_source_bytes(source)
                    source_path = root / f"{variant}.json"
                    source_path.write_bytes(source_bytes)
                    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
                    with self.assertRaisesRegex(ValueError, "ZIP entries|noncanonical entry|canonical NPY|bounded NPY"):
                        load_pr_history_v5_artifact(
                            source_path,
                            bundle,
                            expected_source_file_sha256=source_sha256,
                            expected_bundle_file_sha256=bundle_sha256,
                            expected_history_chain_sha256=self.chain.chain_sha256,
                            expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
                            max_chain_transitions=2,
                        )

    def test_rejects_zip_prefix_trailer_and_attacker_declared_extra_array(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")
            original = artifact.bundle_path.read_bytes()
            for variant, bundle_bytes in (
                ("prefix", b"UNAUTHENTICATED-PREFIX" + original),
                ("trailer", original + b"UNAUTHENTICATED-TRAILER"),
            ):
                with self.subTest(variant=variant):
                    paths = self._retarget_bundle(root, artifact, variant, bundle_bytes)
                    with self.assertRaisesRegex(ValueError, "complete canonical ZIP"):
                        self._load_paths(*paths)

            with zipfile.ZipFile(artifact.bundle_path, "r") as archive:
                entries = [(info.filename, archive.read(info)) for info in archive.infolist()]
            extra = np.array([2901], dtype=np.int64)
            entries.append(("extra/unconsumed_payload.npy", _npy_bytes(extra)))
            extra_bundle = root / "extra.npz"
            self._write_canonical_zip(extra_bundle, entries)
            extra_bundle_sha256 = hashlib.sha256(extra_bundle.read_bytes()).hexdigest()
            source = json.loads(artifact.source_path.read_bytes())
            source["bundle_sha256"] = extra_bundle_sha256
            source["arrays"]["extra/unconsumed_payload"] = {
                "dtype": extra.dtype.str,
                "shape": list(extra.shape),
                "sha256": _array_digest(extra),
                "nbytes": extra.nbytes,
            }
            source_path, source_sha256 = self._write_forged_source(root, "extra", source)
            with self.assertRaisesRegex(ValueError, "exact chain-derived inventory"):
                self._load_paths(source_path, source_sha256, extra_bundle, extra_bundle_sha256)

    def test_rejects_npy_shape_bomb_before_numeric_allocation(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")
            with zipfile.ZipFile(artifact.bundle_path, "r") as archive:
                entries = [(info.filename, archive.read(info)) for info in archive.infolist()]
            first_name, _ = entries[0]
            header = io.BytesIO()
            np.lib.format.write_array_header_2_0(
                header,
                {"descr": np.dtype(np.int64).str, "fortran_order": False, "shape": (10**12,)},
            )
            entries[0] = (first_name, header.getvalue())
            bomb_bundle = root / "shape-bomb.npz"
            self._write_canonical_zip(bomb_bundle, entries)
            bomb_bundle_sha256 = hashlib.sha256(bomb_bundle.read_bytes()).hexdigest()
            source = json.loads(artifact.source_path.read_bytes())
            source["bundle_sha256"] = bomb_bundle_sha256
            first_key = first_name.removesuffix(".npy")
            source["arrays"][first_key].update(
                {"dtype": np.dtype(np.int64).str, "shape": [10**12], "nbytes": 8 * 10**12, "sha256": "0" * 64}
            )
            source_path, source_sha256 = self._write_forged_source(root, "shape-bomb", source)
            max_uncompressed = artifact.bundle_path.stat().st_size * 4
            with unittest.mock.patch(
                "research.principal_stretch.v5_pr_history_artifact.np.frombuffer",
                side_effect=AssertionError("numeric allocation attempted before header bounds"),
            ):
                with self.assertRaisesRegex(ValueError, "oversized individual dimension|shape exceeds"):
                    self._load_paths(
                        source_path,
                        source_sha256,
                        bomb_bundle,
                        bomb_bundle_sha256,
                        max_uncompressed_bytes=max_uncompressed,
                    )

    def test_rejects_unconsumed_nested_source_fields(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")
            for variant in ("sample", "uri"):
                with self.subTest(variant=variant):
                    source = json.loads(artifact.source_path.read_bytes())
                    if variant == "sample":
                        source["samples"][0]["unconsumed_secret"] = {"heldout": "payload"}
                    else:
                        source["artifact_uris"]["unconsumed"] = "secret://payload"
                    source_path, source_sha256 = self._write_forged_source(root, variant, source)
                    with self.assertRaisesRegex(ValueError, "noncanonical keys"):
                        self._load_paths(
                            source_path,
                            source_sha256,
                            artifact.bundle_path,
                            artifact.bundle_file_sha256,
                        )

    def test_rejects_json_numeric_type_aliases(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")

            def shape_float(source):
                shape = source["arrays"]["base_scene/rest_q"]["shape"]
                shape[1] = float(shape[1])

            def schema_float(source):
                source["schema_version"] = 3.0

            def npy_version_float(source):
                source["npy_version"] = [2.0, 0.0]

            def zip_timestamp_float(source):
                source["zip"]["timestamp"] = [float(value) for value in source["zip"]["timestamp"]]

            def trust_bool_as_int(source):
                source["trust_scope"]["persisted_acceptance_is_authority"] = 0

            def reconstructed_sample_shape_float(source):
                shape = source["samples"][0]["operator_geometry"]["source_tet_poses_shape"]
                shape[0] = float(shape[0])

            variants = (
                ("shape-float", shape_float, "noncanonical shape"),
                ("schema-float", schema_float, "unsupported contract"),
                ("npy-version-float", npy_version_float, "bundle writer contract"),
                ("zip-timestamp-float", zip_timestamp_float, "bundle writer contract"),
                ("trust-bool-as-int", trust_bool_as_int, "trust scope"),
                ("sample-shape-float", reconstructed_sample_shape_float, "operator geometry"),
            )
            for variant, mutate, message in variants:
                with self.subTest(variant=variant):
                    source = json.loads(artifact.source_path.read_bytes())
                    mutate(source)
                    source_path, source_sha256 = self._write_forged_source(root, variant, source)
                    with self.assertRaisesRegex(ValueError, message):
                        self._load_paths(
                            source_path,
                            source_sha256,
                            artifact.bundle_path,
                            artifact.bundle_file_sha256,
                        )

            unit_array = np.array([2901], dtype=np.int64)
            bool_shape_record = {
                "dtype": unit_array.dtype.str,
                "shape": [True],
                "sha256": _array_digest(unit_array),
                "nbytes": unit_array.nbytes,
            }
            with self.assertRaisesRegex(ValueError, "noncanonical shape"):
                _safe_npy_array(
                    _npy_bytes(unit_array),
                    bool_shape_record,
                    np.dtype(np.int64),
                    1,
                    key="unittest/bool-shape",
                    max_array_bytes=4096,
                )

    def test_loaded_artifact_aggregate_rejects_trainer_payload_and_identity_tamper(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")
            loaded = self._load(artifact)
            reference = loaded.loaded_samples[0].training_sample.producer_attested_reference_positions
            original = reference.clone()
            reference[0, 0] += 1.0e-6
            try:
                with self.assertRaisesRegex(ValueError, "reference positions changed"):
                    loaded.validate_immutable()
            finally:
                reference.copy_(original)
            original_chain = loaded.source_chain_sha256
            object.__setattr__(loaded, "source_chain_sha256", "0" * 64)
            try:
                with self.assertRaisesRegex(ValueError, "source identities|evidence changed"):
                    loaded.validate_immutable()
            finally:
                object.__setattr__(loaded, "source_chain_sha256", original_chain)
            loaded.validate_immutable()

    def test_persisted_acceptance_is_recomputed_not_trusted(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            artifact = self._write(root, "artifact")
            source = json.loads(artifact.source_path.read_bytes())
            acceptance = source["samples"][0]["reference_acceptance"]
            acceptance["metrics"]["source_float64_accepted_reference"]["gradient_norm"] *= 2.0
            acceptance_payload = dict(acceptance)
            acceptance_payload.pop("acceptance_sha256")
            acceptance["acceptance_sha256"] = canonical_json_sha256(acceptance_payload)
            source["selection"]["selected_reference_acceptance_sha256"][0] = acceptance["acceptance_sha256"]
            selection_payload = dict(source["selection"])
            selection_payload.pop("selection_sha256")
            source["selection"]["selection_sha256"] = canonical_json_sha256(selection_payload)
            source_payload = dict(source)
            source_payload.pop("source_record_sha256")
            source["source_record_sha256"] = canonical_json_sha256(source_payload)
            forged_bytes = _canonical_source_bytes(source)
            forged_path = root / "forged.json"
            forged_path.write_bytes(forged_bytes)
            forged_sha256 = hashlib.sha256(forged_bytes).hexdigest()
            with self.assertRaisesRegex(ValueError, "reference acceptance differs"):
                load_pr_history_v5_artifact(
                    forged_path,
                    artifact.bundle_path,
                    expected_source_file_sha256=forged_sha256,
                    expected_bundle_file_sha256=artifact.bundle_file_sha256,
                    expected_history_chain_sha256=self.chain.chain_sha256,
                    expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
                    max_chain_transitions=2,
                )


if __name__ == "__main__":
    unittest.main()
