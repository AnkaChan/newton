# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic reference-sequence shard loading."""

from __future__ import annotations

import collections
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from .. import reference_rollout as rollout_module
from ..reference_rollout import (
    FreeBodyReferenceProtocol,
    FreeBodyReferenceScene,
    ReferenceCandidateEvidence,
    ReferenceRollout,
    ReferenceStepEvidence,
    write_reference_rollout_shard,
)
from ..reference_sequence_dataset import (
    REFERENCE_EXECUTION_DT_SECONDS,
    ReferenceSequenceDataset,
    ReferenceTransitionKey,
    canonical_reference_state_float64_sha256,
    reference_sequence_index_header,
)
from ..solver_benchmark import CommonStateMetrics
from ..v5_dataset import DatasetRole


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _static_arrays(offset: float = 0.0) -> dict[str, np.ndarray]:
    staged_rest_q = np.asarray(
        (
            (offset + 0.0, 0.0, 0.0),
            (offset + 1.0, 0.0, 0.0),
            (offset + 0.0, 1.0, 0.0),
            (offset + 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    return {
        "rest_q": staged_rest_q.astype(np.float64),
        "tet_indices": np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        "tet_poses": np.eye(3, dtype=np.float64)[None, :, :],
        "mass": np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32),
        "particle_inv_mass": np.asarray((1.0, 0.5, 1.0 / 3.0, 0.25), dtype=np.float32),
        "particle_flags": np.ones((4,), dtype=np.int32),
        "tet_materials": np.asarray(((4.0, 7.0, 0.0),), dtype=np.float32),
        "boundary_triangles": np.asarray(
            ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
            dtype=np.int32,
        ),
        "color_group_offsets": np.asarray((0, 4), dtype=np.int32),
        "color_group_particles": np.arange(4, dtype=np.int32),
        "hierarchy_tet_adj": np.full((1, 4), -1, dtype=np.int32),
        "hierarchy_tet_c0": np.asarray(((offset + 0.25, 0.25, 0.25),), dtype=np.float64),
        "hierarchy_tet_vol": np.asarray((1.0 / 6.0,), dtype=np.float64),
        "hierarchy_level_0_adj": np.full((1, 1), -1, dtype=np.int32),
        "hierarchy_level_0_assign": np.asarray((0,), dtype=np.int32),
        "hierarchy_level_0_c0": np.asarray(((offset + 0.25, 0.25, 0.25),), dtype=np.float64),
        "hierarchy_level_0_pou_idx": np.asarray(((0,),), dtype=np.int32),
        "hierarchy_level_0_pou_w": np.asarray(((1.0,),), dtype=np.float64),
        "hierarchy_level_0_vol": np.asarray((1.0 / 6.0,), dtype=np.float64),
    }


def _sequence_arrays(rest_q: np.ndarray, *, step_count: int = 3) -> dict[str, np.ndarray]:
    dt = REFERENCE_EXECUTION_DT_SECONDS
    staged_q = np.empty((step_count + 1, *rest_q.shape), dtype=np.float32)
    staged_qd = np.empty_like(staged_q)
    vertex_scale = np.arange(rest_q.shape[0], dtype=np.float32)[:, None]
    direction = np.asarray((0.01, -0.02, 0.03), dtype=np.float32)[None, :]
    for frame in range(step_count + 1):
        staged_q[frame] = rest_q + np.float32(frame) * vertex_scale * direction
    staged_qd[0] = np.asarray((0.2, -0.1, 0.05), dtype=np.float32)
    for frame in range(1, step_count + 1):
        staged_qd[frame] = np.asarray(
            (staged_q[frame] - staged_q[frame - 1]) / dt,
            dtype=np.float32,
        )
    staged_inertial_target = np.add(
        staged_q[:-1],
        np.multiply(dt, staged_qd[:-1], dtype=np.float32),
        dtype=np.float32,
    )
    return {
        "q": staged_q.astype(np.float64),
        "qd": staged_qd.astype(np.float64),
        "inertial_target": staged_inertial_target.astype(np.float64),
        "external_force": np.zeros((step_count, *rest_q.shape), dtype=np.float32),
        "pinned_indices": np.empty((0,), dtype=np.int32),
        "pin_targets": np.empty((step_count, 0, 3), dtype=np.float64),
    }


def _metrics(positions: np.ndarray, *, objective: float, residual: float) -> CommonStateMetrics:
    return CommonStateMetrics(
        objective=objective,
        inertia=0.25 * objective,
        elastic=0.75 * objective,
        gradient_norm=residual,
        relative_residual=residual,
        determinant_min=0.8,
        determinant_max=1.2,
        inverted_tet_fraction=0.0,
        minimum_singular_value=0.7,
        free_rms_error_m=None,
        mass_weighted_rms_error_m=None,
        max_pin_error_m=0.0,
        position_sha256=rollout_module._array_digest(positions),
    )


def _self_hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = rollout_module._canonical_json_digest(payload)
    return result


def _accepted_rollout(
    *,
    asset_id: str,
    source_label: str,
    static: dict[str, np.ndarray],
    sequence: dict[str, np.ndarray],
) -> ReferenceRollout:
    step_count = sequence["q"].shape[0] - 1
    protocol = FreeBodyReferenceProtocol(rollout_steps=step_count)
    initial_scene = _self_hashed({"contract": "synthetic-initial-scene-v1", "asset_id": asset_id}, "scene_sha256")
    template = types.SimpleNamespace(
        x_current=sequence["q"][0],
        velocity=sequence["qd"][0],
        manifest=lambda: dict(initial_scene),
    )
    scene = types.SimpleNamespace(
        template=template,
        initial_state=types.SimpleNamespace(
            source_center=np.zeros(3, dtype=np.float64),
            source_characteristic_length=1.0,
            normalized_characteristic_length_m=1.0,
            orientation_repaired_count=0,
        ),
        protocol=protocol,
        asset_id=asset_id,
        source=f"{source_label}.vtk",
        source_sha256=hashlib.sha256(f"source:{source_label}".encode()).hexdigest(),
        deformation_seed=101,
        velocity_seed=211,
        static_arrays=lambda: static,
    )
    scene.physical_identities = lambda: FreeBodyReferenceScene.physical_identities(scene)

    steps: list[ReferenceStepEvidence] = []
    for step_id in range(step_count):
        dynamic_arrays = {
            "external_force": rollout_module._array_record(sequence["external_force"][step_id]),
            "pin_targets": rollout_module._array_record(sequence["pin_targets"][step_id]),
            "pinned_indices": rollout_module._array_record(sequence["pinned_indices"]),
            "vbd_inertial_target": rollout_module._array_record(sequence["inertial_target"][step_id]),
            "velocity": rollout_module._array_record(sequence["qd"][step_id]),
            "x_current": rollout_module._array_record(sequence["q"][step_id]),
        }
        dynamic_scene = _self_hashed(
            {"contract": "synthetic-dynamic-scene-v1", "arrays": dynamic_arrays},
            "scene_sha256",
        )
        objective = _self_hashed(
            {"contract": "synthetic-objective-v1", "scene_sha256": dynamic_scene["scene_sha256"]},
            "objective_instance_sha256",
        )
        output_positions = sequence["q"][step_id + 1]
        output_velocity = sequence["qd"][step_id + 1]
        candidates = tuple(
            ReferenceCandidateEvidence(
                iterations=iterations,
                effective_tile_solve=False,
                metrics=_metrics(output_positions, objective=objective_value, residual=residual),
                position_float32_sha256=rollout_module._array_digest(output_positions.astype(np.float32)),
                velocity_float32_sha256=rollout_module._array_digest(output_velocity.astype(np.float32)),
                velocity_float64_sha256=rollout_module._array_digest(output_velocity),
                displacement_from_previous_budget_m=None if iterations == 20 else 0.0,
                relative_residual_over_iterate_zero=residual,
            )
            for iterations, objective_value, residual in ((20, 9.0, 0.1), (50, 8.0, 0.04), (100, 7.0, 0.01))
        )
        steps.append(
            ReferenceStepEvidence(
                step_id=step_id,
                input_position_sha256=rollout_module._array_digest(sequence["q"][step_id]),
                input_velocity_sha256=rollout_module._array_digest(sequence["qd"][step_id]),
                inertial_target_sha256=rollout_module._array_digest(sequence["inertial_target"][step_id]),
                dynamic_scene_manifest=dynamic_scene,
                objective_manifest=objective,
                dynamic_scene_sha256=dynamic_scene["scene_sha256"],
                objective_instance_sha256=objective["objective_instance_sha256"],
                iterate_zero_metrics=_metrics(sequence["inertial_target"][step_id], objective=10.0, residual=1.0),
                candidates=candidates,
                selected_iterations=100,
                output_position_sha256=rollout_module._array_digest(output_positions),
                output_velocity_sha256=rollout_module._array_digest(output_velocity),
                exact_velocity_commit=True,
                reference_accepted=True,
                reference_failures=(),
            )
        )
    return ReferenceRollout(
        scene=scene,
        q=sequence["q"],
        qd=sequence["qd"],
        inertial_target=sequence["inertial_target"],
        external_force=sequence["external_force"],
        pinned_indices=sequence["pinned_indices"],
        pin_targets=sequence["pin_targets"],
        steps=tuple(steps),
    )


def _write_sequence_record(
    root: Path,
    *,
    role: DatasetRole,
    asset_id: str,
    sequence_id: str,
    offset: float = 0.0,
    source_label: str | None = None,
    inertial_target_offset: float = 0.0,
) -> dict[str, object]:
    static = _static_arrays(offset)
    sequence = _sequence_arrays(static["rest_q"])
    sequence["inertial_target"] = np.add(
        sequence["inertial_target"].astype(np.float32),
        np.float32(inertial_target_offset),
        dtype=np.float32,
    ).astype(np.float64)
    rollout = _accepted_rollout(
        asset_id=asset_id,
        source_label=asset_id if source_label is None else source_label,
        static=static,
        sequence=sequence,
    )
    files = write_reference_rollout_shard(root / asset_id, rollout, sequence_id=sequence_id)
    manifest = json.loads(files.manifest_json.read_text(encoding="utf-8"))
    identities = manifest["identities"]
    return {
        "role": role.value,
        "asset_id": asset_id,
        "asset_source_sha256": manifest["source_sha256"],
        "sequence_id": sequence_id,
        "topology_sha256": identities["topology_sha256"],
        "operator_sha256": identities["operator_sha256"],
        "material_sha256": identities["material_sha256"],
        "protocol_sha256": identities["protocol_sha256"],
        "producer_manifest_json": files.manifest_json.relative_to(root).as_posix(),
        "producer_manifest_json_sha256": _file_sha256(files.manifest_json),
        "step_ids": list(range(sequence["q"].shape[0] - 1)),
        "reference_state_float64_sha256": [
            canonical_reference_state_float64_sha256(sequence["q"][step_id + 1])
            for step_id in range(sequence["q"].shape[0] - 1)
        ],
    }


def _write_index(root: Path, records: list[dict[str, object]], name: str = "index.json") -> Path:
    payload = reference_sequence_index_header()
    payload["splits"] = {role.value: [] for role in DatasetRole}
    for record in records:
        payload["splits"][record["role"]].append(record)
    index_path = root / name
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return index_path


def _rewrite_sequence_npz(root: Path, record: dict[str, object], mutate) -> None:
    manifest_path = root / record["producer_manifest_json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence_record = manifest["files"]["sequence_npz"]
    sequence_path = manifest_path.parent / sequence_record["path"]
    with np.load(sequence_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(arrays)
    contents = rollout_module._npz_bytes(arrays)
    sequence_path.write_bytes(contents)
    sequence_record.update(
        {
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "arrays": {name: rollout_module._array_record(value) for name, value in sorted(arrays.items())},
        }
    )
    manifest["inventory_sha256"] = rollout_module._canonical_json_digest(
        {"files": manifest["files"], "identities": manifest["identities"]}
    )
    manifest_path.write_bytes(rollout_module._json_bytes(manifest))
    record["producer_manifest_json_sha256"] = _file_sha256(manifest_path)


class TestReferenceSequenceDataset(unittest.TestCase):
    def test_writer_integration_transition_and_stateless_sampling_preserve_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                _write_sequence_record(
                    root, role=DatasetRole.TRAIN, asset_id="zeta", sequence_id="sample-001", offset=3.0
                ),
                _write_sequence_record(
                    root,
                    role=DatasetRole.VALIDATION,
                    asset_id="held-out",
                    sequence_id="sample-000",
                    offset=6.0,
                ),
                _write_sequence_record(root, role=DatasetRole.TRAIN, asset_id="alpha", sequence_id="sample-000"),
                _write_sequence_record(
                    root, role=DatasetRole.TRAIN, asset_id="zeta", sequence_id="sample-000", offset=3.0
                ),
            ]
            index_path = _write_index(root, records)
            reversed_index_path = _write_index(root, list(reversed(records)), "index-reversed.json")

            dataset = ReferenceSequenceDataset.load(index_path)
            reordered = ReferenceSequenceDataset.load(reversed_index_path)
            self.assertEqual(
                tuple((record.asset_id, record.sequence_id) for record in dataset.records(DatasetRole.TRAIN)),
                (("alpha", "sample-000"), ("zeta", "sample-000"), ("zeta", "sample-001")),
            )
            self.assertEqual(dataset.index_sha256, reordered.index_sha256)

            key = ReferenceTransitionKey(asset_id="zeta", sequence_id="sample-001", step_id=1)
            transition = dataset.transition(key)
            manifest = json.loads((root / records[0]["producer_manifest_json"]).read_text(encoding="utf-8"))
            sequence_path = root / "zeta" / manifest["files"]["sequence_npz"]["path"]
            with np.load(sequence_path, allow_pickle=False) as source:
                expected_previous = (
                    source["q"][1].astype(np.float32) - source["dt"] * source["qd"][1].astype(np.float32)
                ).astype(np.float64)
                expected_reference = source["q"][2]
                np.testing.assert_array_equal(transition.velocity, source["qd"][1])
            self.assertEqual(transition.key, key)
            self.assertEqual(transition.role, DatasetRole.TRAIN)
            self.assertEqual(transition.topology_sha256, records[0]["topology_sha256"])
            self.assertEqual(transition.operator_sha256, records[0]["operator_sha256"])
            self.assertEqual(transition.material_sha256, records[0]["material_sha256"])
            self.assertEqual(transition.protocol_sha256, records[0]["protocol_sha256"])
            np.testing.assert_array_equal(transition.x_previous, expected_previous)
            np.testing.assert_array_equal(transition.reference_positions, expected_reference)
            self.assertEqual(
                transition.reference_state_float64_sha256,
                canonical_reference_state_float64_sha256(expected_reference),
            )
            self.assertFalse(transition.reference_positions.flags["W"])
            self.assertEqual(transition.static.rest_q.dtype, np.float64)
            self.assertIn("hierarchy_level_0_pou_w", transition.static.hierarchy_arrays)
            self.assertEqual(transition.pinned_indices.shape, (0,))
            self.assertEqual(transition.pin_targets.shape, (0, 3))
            self.assertEqual(manifest["files"]["sequence_npz"]["arrays"]["pin_targets"]["nbytes"], 0)

            first = dataset.sample_keys(DatasetRole.TRAIN, count=10, seed=179)
            self.assertEqual(first, dataset.sample_keys("train", count=10, seed=179))
            self.assertEqual(first, reordered.sample_keys(DatasetRole.TRAIN, count=10, seed=179))
            self.assertEqual(collections.Counter(key.asset_id for key in first), {"alpha": 5, "zeta": 5})

    def test_index_rejects_role_aliases_and_asset_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_sequence_record(
                root, role=DatasetRole.TRAIN, asset_id="shared-asset", sequence_id="sample-000"
            )
            validation_same_asset = _write_sequence_record(
                root,
                role=DatasetRole.VALIDATION,
                asset_id="shared-asset",
                sequence_id="sample-001",
            )
            validation_same_source = _write_sequence_record(
                root,
                role=DatasetRole.VALIDATION,
                asset_id="different-name",
                sequence_id="sample-000",
                offset=3.0,
                source_label="shared-asset",
            )

            alias_path = _write_index(root, [train], "role-alias.json")
            alias_payload = json.loads(alias_path.read_text(encoding="utf-8"))
            alias_payload["splits"]["val"] = alias_payload["splits"].pop("validation")
            alias_path.write_text(json.dumps(alias_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split roles must be exactly"):
                ReferenceSequenceDataset.load(alias_path)

            with self.assertRaisesRegex(ValueError, "asset_id appears in multiple roles"):
                ReferenceSequenceDataset.load(_write_index(root, [train, validation_same_asset], "reused-id.json"))
            with self.assertRaisesRegex(ValueError, "asset_source_sha256 appears in multiple roles"):
                ReferenceSequenceDataset.load(_write_index(root, [train, validation_same_source], "reused-source.json"))

    def test_payload_rejects_wrong_dt_nonpromotion_and_reference_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_dt = _write_sequence_record(
                root, role=DatasetRole.TRAIN, asset_id="wrong-dt", sequence_id="sample-000"
            )
            _rewrite_sequence_npz(
                root,
                wrong_dt,
                lambda arrays: arrays.__setitem__("dt", np.asarray(np.float32(1.0 / 240.0))),
            )
            wrong_dt_dataset = ReferenceSequenceDataset.load(_write_index(root, [wrong_dt], "wrong-dt.json"))
            with self.assertRaisesRegex(ValueError, "dt must equal the reference execution dt"):
                wrong_dt_dataset.transition(
                    ReferenceTransitionKey(asset_id="wrong-dt", sequence_id="sample-000", step_id=0)
                )

            nonpromotion = _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="nonpromotion",
                sequence_id="sample-000",
                offset=3.0,
            )

            def perturb_q(arrays):
                arrays["q"][1, 0, 0] = np.nextafter(arrays["q"][1, 0, 0], np.inf)

            _rewrite_sequence_npz(root, nonpromotion, perturb_q)
            nonpromotion_dataset = ReferenceSequenceDataset.load(
                _write_index(root, [nonpromotion], "nonpromotion.json")
            )
            with self.assertRaisesRegex(ValueError, "lossless float64 promotion"):
                nonpromotion_dataset.transition(
                    ReferenceTransitionKey(asset_id="nonpromotion", sequence_id="sample-000", step_id=0)
                )

            wrong_hash = _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="wrong-hash",
                sequence_id="sample-000",
                offset=6.0,
            )
            wrong_hash["reference_state_float64_sha256"][1] = hashlib.sha256(b"not-the-reference").hexdigest()
            wrong_hash_dataset = ReferenceSequenceDataset.load(_write_index(root, [wrong_hash], "wrong-hash.json"))
            with self.assertRaisesRegex(ValueError, "float64 reference-state SHA-256 mismatch"):
                wrong_hash_dataset.transition(
                    ReferenceTransitionKey(asset_id="wrong-hash", sequence_id="sample-000", step_id=1)
                )

    def test_payload_rejects_physically_wrong_but_self_consistent_inertial_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="wrong-inertial-target",
                sequence_id="sample-000",
                inertial_target_offset=0.125,
            )
            dataset = ReferenceSequenceDataset.load(_write_index(root, [record]))
            with self.assertRaisesRegex(ValueError, "inertial_target does not match"):
                dataset.transition(
                    ReferenceTransitionKey(
                        asset_id="wrong-inertial-target",
                        sequence_id="sample-000",
                        step_id=0,
                    )
                )


if __name__ == "__main__":
    unittest.main()
