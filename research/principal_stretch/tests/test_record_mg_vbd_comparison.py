# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from .. import pr_scene_history
from .. import record_mg_vbd_comparison as recording
from ..captured_graph_vbd import REASON_NAMES
from ..solver_benchmark import run_newton


def _seal_manifest(payload: dict[str, object], digest_name: str) -> dict[str, object]:
    result = copy.deepcopy(payload)
    result[digest_name] = hashlib.sha256(recording._canonical_json(result)).hexdigest()
    return result


def _fake_generation_source() -> dict[str, object]:
    files = {
        name: {
            "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
            "git_blob_oid": hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest(),
        }
        for name in recording._GENERATION_SOURCE_PATHS
    }
    return _seal_manifest(
        {
            "contract": recording.GENERATION_SOURCE_SCHEMA,
            "git_revision": "a" * 40,
            "git_tree_oid": "b" * 40,
            "git_object_format": "sha1",
            "repository_clean": True,
            "repository_status_sha256": hashlib.sha256(b"").hexdigest(),
            "files": files,
            "newton_version": "unit-test-newton",
            "warp_version": "unit-test-warp",
        },
        "manifest_sha256",
    )


def _valid_bundle(
    scene_key: str = "twist",
    *,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    spec = recording.recording_spec(scene_key)
    scene = recording.build_recording_scene(scene_key, gaia_asset_root=gaia_asset_root)
    stored_frames = 2
    state_shape = (len(recording.METHOD_IDS), stored_frames, scene.n_vertices, 3)
    positions = np.broadcast_to(scene.x_current, state_shape).copy()
    velocities = np.broadcast_to(scene.velocity, state_shape).copy()
    pin_targets = np.stack(
        (
            recording.pin_targets_for_frame(scene_key, scene, 0),
            recording.pin_targets_for_frame(scene_key, scene, 0),
        )
    )
    positions[:, :, scene.pinned_indices] = pin_targets[None, :, :, :]
    velocities[:, :, scene.pinned_indices] = 0.0
    objective_positions = positions.copy()
    objective_velocities = velocities.copy()
    arrays: dict[str, np.ndarray] = {
        "positions": positions,
        "velocities": velocities,
        "objective_input_positions": objective_positions,
        "objective_input_velocities": objective_velocities,
        "pin_targets": pin_targets,
        "time_seconds": np.array([0.0, spec.substeps_per_source_frame * scene.dt], dtype=np.float64),
        "source_frame_index": np.array([-1, 0], dtype=np.int64),
        "solve_seconds": np.zeros((3, stored_frames), dtype=np.float64),
        "transfer_seconds": np.zeros((3, stored_frames), dtype=np.float64),
        "mg_last_gate_accepted": np.array([[-1] * 4, [1] * 4], dtype=np.int8),
        "mg_last_gate_reason_code": np.array([[-1] * 4, [REASON_NAMES.index("accepted")] * 4], dtype=np.int16),
        "mg_frame_gate_accept_count": np.array([0, 4], dtype=np.int64),
    }
    arrays.update(recording._empty_metric_arrays(stored_frames))
    for method in range(3):
        objective_scene = dataclasses.replace(
            scene,
            x_current=objective_positions[method, 1],
            velocity=objective_velocities[method, 1],
            pin_targets=pin_targets[1],
        )
        metrics = recording.evaluate_common_state(
            recording.build_common_problem(objective_scene),
            positions[method, 1],
            reference_positions=positions[0, 1],
        )
        recording._store_metrics(arrays, method, 1, metrics)

    generation_source = _fake_generation_source()
    metadata: dict[str, object] = {
        "schema": recording.SCHEMA,
        "scene_key": scene_key,
        "scene_display_name": spec.display_name,
        "scene_manifest": recording._historical_scene_manifest(scene, generation_source["git_revision"]),
        "scene_physical_sha256": recording._scene_physical_sha256(scene),
        "git_revision": generation_source["git_revision"],
        "generation_source": generation_source,
        "methods": recording._method_records(spec),
        "method_order": list(recording.METHOD_IDS),
        "reference_policy": recording._reference_policy(spec),
        "simulation": {
            "source_frames": 1,
            "stored_frames_including_initial": stored_frames,
            "source_frame_rate_hz": 60,
            "substeps_per_source_frame": spec.substeps_per_source_frame,
            "atomic_dt_seconds": scene.dt,
            "stored_duration_seconds": float(arrays["time_seconds"][-1]),
            "source_schedule": recording._source_schedule(scene_key),
        },
        "metrics": recording._metrics_policy(),
        "mg_gate_reason_names": list(REASON_NAMES),
        "setup_seconds_diagnostic": {
            "reference_public_vbd": 0.0,
            "mg_vbd_capture_and_setup": 0.0,
            "vbd_k4_public": 0.0,
        },
        "device": {"requested": "cuda:0", "resolved": "cuda:0", "is_cuda": True, "name": "unit-test CUDA"},
        "camera": recording.fixed_camera(
            positions,
            panel_width=640,
            panel_height=720,
            direction=spec.camera_direction,
        ),
        "static_first_step_newton_reference": None,
        "gaia_asset_bundle": recording._gaia_asset_bundle_manifest(scene_key),
        "execution_authentication": recording._execution_authentication_policy(),
    }
    return metadata, arrays


def _write_unvalidated_bundle(
    directory: pathlib.Path,
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> pathlib.Path:
    npz_path = directory / "raw.npz"
    np.savez_compressed(npz_path, **arrays)
    npz_sha256 = recording._file_sha256(npz_path)
    addressed_npz = directory / f"{metadata['scene_key']}-{npz_sha256}.npz"
    npz_path.replace(addressed_npz)
    payload = copy.deepcopy(metadata)
    payload.update(
        {
            "npz_filename": addressed_npz.name,
            "npz_file_sha256": npz_sha256,
            "arrays": {
                name: {
                    "dtype": value.dtype.name,
                    "shape": list(value.shape),
                    "array_sha256": recording.array_sha256(value),
                }
                for name, value in arrays.items()
            },
        }
    )
    record_sha256 = hashlib.sha256(recording._canonical_json(payload)).hexdigest()
    payload["record_sha256"] = record_sha256
    path = directory / f"{metadata['scene_key']}-{record_sha256}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _save_validated_bundle(
    directory: str,
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
    *,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    with (
        mock.patch.object(recording, "_generation_source_manifest", return_value=metadata["generation_source"]),
        mock.patch.object(recording, "_consume_generated_trajectory"),
        mock.patch.object(recording, "_finalize_generated_trajectory"),
    ):
        return recording.save_content_addressed_bundle(
            directory,
            metadata,
            arrays,
            gaia_asset_root=gaia_asset_root,
        )


def _load_test_bundle(path: pathlib.Path, *, gaia_asset_root: str | os.PathLike[str] | None = None):
    with mock.patch.object(recording, "_verify_generation_source_git_objects"):
        return recording.load_content_addressed_bundle(path, gaia_asset_root=gaia_asset_root)


def _fake_render_source() -> dict[str, object]:
    return _seal_manifest(
        {
            "contract": recording.RENDER_SOURCE_SCHEMA,
            "files": {
                "research/principal_stretch/record_mg_vbd_comparison.py": "1" * 64,
                "newton_capture/__init__.py": "2" * 64,
                "newton_capture/_deps.py": "3" * 64,
                "newton_capture/_display.py": "4" * 64,
                "newton_capture/_video.py": "5" * 64,
            },
            "newton_version": "unit-test-newton",
            "warp_version": "unit-test-warp",
            "pillow_version": "unit-test-pillow",
            "imageio_version": "unit-test-imageio",
            "imageio_ffmpeg_version": "unit-test-imageio-ffmpeg",
        },
        "manifest_sha256",
    )


def _git(repository: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _generation_repository(directory: str) -> pathlib.Path:
    repository = pathlib.Path(directory)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Recorder Test")
    _git(repository, "config", "user.email", "recorder@example.invalid")
    for index, relative in enumerate(recording._GENERATION_SOURCE_PATHS):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# executed source {index}: {relative}\n", encoding="utf-8")
    _git(repository, "add", "--", *recording._GENERATION_SOURCE_PATHS)
    _git(repository, "commit", "--quiet", "-m", "Create executed source closure")
    return repository


class TestRecordingPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.medium = recording.build_recording_scene("refinement-medium")
        cls.twist = recording.build_recording_scene("twist")

    def test_two_proof_scenes_use_audited_pr_schedules(self):
        medium = recording.recording_spec("refinement-medium")
        twist = recording.recording_spec("twist")
        self.assertEqual(medium.substeps_per_source_frame, 6)
        self.assertEqual(medium.reference_iterations, 100)
        self.assertEqual(twist.substeps_per_source_frame, 5)
        self.assertEqual(twist.reference_iterations, 20)
        self.assertEqual(self.medium.n_vertices, 525)
        self.assertEqual(self.medium.n_tets, 1600)
        self.assertEqual(self.twist.n_vertices, 272)
        self.assertEqual(self.twist.n_tets, 720)

    def test_gaia_specs_route_only_the_two_pinned_assets(self):
        expected = {
            "gaia-bunny-small": ("bunny_small", 0.1, "Gaia bunny_small"),
            "gaia-armadilo-lowres": ("Armadilo_lowres", 1.0, "Gaia Armadilo_lowres"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for scene_key, (asset_name, scale, display_prefix) in expected.items():
                with self.subTest(scene=scene_key):
                    spec = recording.recording_spec(scene_key)
                    self.assertEqual(spec.substeps_per_source_frame, 6)
                    self.assertEqual(spec.reference_iterations, 40)
                    self.assertTrue(spec.display_name.startswith(display_prefix))
                    marker = object()
                    with mock.patch.object(recording, "build_registered_gaia_scene", return_value=marker) as build:
                        actual = recording.build_recording_scene(scene_key, gaia_asset_root=directory)
                    self.assertIs(actual, marker)
                    build.assert_called_once_with(
                        asset_name,
                        pathlib.Path(directory).resolve(),
                        unit_scale_m_per_source_unit=scale,
                    )

    def test_gaia_scene_requires_an_explicit_or_environment_asset_root(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PSS_GAIA_ASSET_ROOT"):
                recording.build_recording_scene("gaia-bunny-small")

        with tempfile.TemporaryDirectory() as directory:
            marker = object()
            with (
                mock.patch.dict(os.environ, {"PSS_GAIA_ASSET_ROOT": directory}),
                mock.patch.object(recording, "build_registered_gaia_scene", return_value=marker) as build,
            ):
                actual = recording.build_recording_scene("gaia-bunny-small")
            self.assertIs(actual, marker)
            self.assertEqual(build.call_args.args[1], pathlib.Path(directory).resolve())

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only for Gaia"):
                recording.build_recording_scene("twist", gaia_asset_root=directory)

    def test_gaia_targets_and_source_schedule_are_static(self):
        scene = mock.Mock()
        scene.pin_targets = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        for scene_key in ("gaia-bunny-small", "gaia-armadilo-lowres"):
            with self.subTest(scene=scene_key):
                first = recording.pin_targets_for_frame(scene_key, scene, 0)
                second = recording.pin_targets_for_frame(scene_key, scene, 149)
                np.testing.assert_array_equal(first, scene.pin_targets)
                np.testing.assert_array_equal(second, scene.pin_targets)
                self.assertFalse(np.shares_memory(first, scene.pin_targets))
                self.assertEqual(recording._source_schedule(scene_key), "fixed support slab under gravity and tip load")

    def test_gaia_protocol_annotation_declares_scale_support_load_and_gravity(self):
        self.assertEqual(
            recording._gaia_protocol_annotation("gaia-bunny-small"),
            "scale 0.1 m/source unit · fixed min-y 2% slab · +x 10 N total on max-y 2% slab · gravity -y 9.81 m/s²",
        )
        self.assertEqual(
            recording._gaia_protocol_annotation("gaia-armadilo-lowres"),
            "scale 1 m/source unit · fixed min-y 2% slab · +x 10 N total on max-y 2% slab · gravity -y 9.81 m/s²",
        )
        self.assertIsNone(recording._gaia_protocol_annotation("twist"))

    def test_gaia_generation_preflights_the_complete_asset_package_before_scene_build(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(recording, "build_recording_scene") as build,
                mock.patch.object(recording, "_PublicVBDTrajectory") as public,
                mock.patch.object(recording, "_MGVBDTrajectory") as mg,
            ):
                with self.assertRaisesRegex(ValueError, "Gaia asset bundle file"):
                    recording.generate_trajectory(
                        "gaia-bunny-small",
                        source_frames=1,
                        device="cpu",
                        gaia_asset_root=directory,
                    )
            build.assert_not_called()
            public.assert_not_called()
            mg.assert_not_called()

    def test_gaia_asset_root_is_available_to_generate_and_render_commands(self):
        parser = recording._build_parser()
        generated = parser.parse_args(
            [
                "generate",
                "--scene",
                "gaia-bunny-small",
                "--frames",
                "1",
                "--out-dir",
                "proof",
                "--gaia-asset-root",
                "gaia",
            ]
        )
        rendered = parser.parse_args(
            [
                "render",
                "--bundle",
                "proof.json",
                "--out",
                "proof.mp4",
                "--gaia-asset-root",
                "gaia",
            ]
        )
        self.assertEqual(generated.gaia_asset_root, "gaia")
        self.assertEqual(rendered.gaia_asset_root, "gaia")

    def test_twist_targets_match_existing_exact_history_callback(self):
        history = pr_scene_history.create_pr_scene_history("twist")
        initial = history.initial_checkpoint.state
        for source_frame in (0, 1, 199, 200, 250):
            state = dataclasses.replace(
                initial,
                coordinate=pr_scene_history.AtomicCoordinate(source_frame, 0),
            )
            expected = history.apply_callback(state).pin_targets
            actual = recording.twist_pin_targets(self.twist, source_frame)
            np.testing.assert_array_equal(actual.astype(np.float32), expected)

    def test_twist_targets_are_constant_data_for_all_substeps(self):
        first = recording.pin_targets_for_frame("twist", self.twist, 17)
        second = recording.pin_targets_for_frame("twist", self.twist, 17)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.shares_memory(first, second))

    def test_camera_is_deterministic_and_uses_all_methods(self):
        positions = np.array(
            [
                [[[-1.0, -0.5, 0.0], [1.0, 0.5, 2.0]]],
                [[[-2.0, -1.0, -1.0], [2.0, 1.0, 3.0]]],
                [[[-1.5, -0.75, -0.5], [1.5, 0.75, 2.5]]],
            ],
            dtype=np.float64,
        )
        first = recording.fixed_camera(
            positions,
            panel_width=640,
            panel_height=720,
            direction=(1.0, -2.0, 0.25),
        )
        second = recording.fixed_camera(
            positions,
            panel_width=640,
            panel_height=720,
            direction=(1.0, -2.0, 0.25),
        )
        self.assertEqual(first, second)
        self.assertEqual(first["union_aabb_min_m"], [-2.0, -1.0, -1.0])
        self.assertEqual(first["union_aabb_max_m"], [2.0, 1.0, 3.0])
        self.assertEqual(first["target"], [0.0, 0.0, 1.0])

    def test_method_records_make_reference_limitations_explicit(self):
        methods = recording._method_records(recording.recording_spec("refinement-medium"))
        self.assertEqual([item["id"] for item in methods], list(recording.METHOD_IDS))
        self.assertEqual([item["panel_title"] for item in methods], ["REFERENCE*", "MG-VBD", "VBD K4"])
        self.assertFalse(methods[0]["ground_truth_claim"])
        self.assertFalse(methods[0]["newton_claim"])
        self.assertEqual(methods[0]["iterations_per_atomic_step"], 100)
        self.assertEqual(methods[2]["iterations_per_atomic_step"], 4)

    def test_gpu_generation_uses_committed_rollout_and_public_vbd_without_manual_claim(self):
        source = inspect.getsource(recording)
        self.assertIn("MGVBDRolloutCapturedBackend.build", source)
        self.assertIn("MGVBDRollout(scene, self.backend)", source)
        self.assertIn("SolverVBD(", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)


@unittest.skipUnless(os.environ.get("PSS_GAIA_ASSET_ROOT"), "set PSS_GAIA_ASSET_ROOT for Gaia recorder tests")
class TestGaiaRecordingAssets(unittest.TestCase):
    def test_gaia_asset_manifest_rejects_json_boolean_numeric_alias(self):
        value = recording._gaia_asset_bundle_manifest("gaia-armadilo-lowres")
        self.assertIsNotNone(value)
        changed = copy.deepcopy(value)
        changed["unit_scale_m_per_source_unit"] = True
        with self.assertRaisesRegex(ValueError, "not canonical"):
            recording._validate_gaia_asset_bundle_manifest(changed, "gaia-armadilo-lowres")

    def test_real_assets_reconstruct_exact_recording_scenes(self):
        root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        expected = {
            "gaia-bunny-small": {
                "counts": (1839, 5891, 3674, 142, 2),
                "source": "5052f098fd0eba9efa20c6dbb4a8915f50df09948a4b9d438a44976e86f9b746",
                "physical": "24f4eddc5500712f66472b176033befa5f6ec29e307bcf3e6015bbdadc15acbe",
                "source_topology": "598c36655ec2eb4071a201be92c2c10d5c1d8d5852edbba893d602c48dd0b3f4",
                "topology": "45fb137b9d9bee8520e9013ce7d461521a49834e0d7f5792f95b17c64aec3bd5",
                "material": "730f434320a69962fe87e2ed347408d43da048e922f3545996626a24099be026",
                "boundary": "30db2b091ebbc1c681d75e6276e26e777369064d5460c9b2e80ce55eccfca27e",
                "mass": "34a5f1f706b69415e467a4362d36674de33292664ab1dd4b179e2ad3b384c0da",
            },
            "gaia-armadilo-lowres": {
                "counts": (3992, 14870, 6000, 116, 17),
                "source": "6226e096aa61f27ec4de582fcf82d834bf2647bbfcbaefb0ba9c320d99809644",
                "physical": "a633c58e2b3b10040402b5f909bb26bce705e658fc793cc1c84f1c9949fbbb21",
                "source_topology": "837b928219aa5283b9167cc751daa35e0dcbc7b40c2cd8c2780781bd69e81046",
                "topology": "217c38210a98851a1205af95f38fdaa150ab24db43d9ca74296d4d885bf0a200",
                "material": "c5e428af9ea9ae8fa2d623b6fada5c5ec1d75f2a5f461143ba2d2ffc4895fa94",
                "boundary": "5ea472aec69867d7abbfcb1296794fee54cee34656a3caf3eb68522719a85421",
                "mass": "577847731bd889da5ad70fc8b10311302e127c1faa3fb6a17ddf6e6114ecd798",
            },
        }
        for scene_key, evidence in expected.items():
            with self.subTest(scene=scene_key):
                first = recording.build_recording_scene(scene_key, gaia_asset_root=root)
                second = recording.build_recording_scene(scene_key, gaia_asset_root=root)
                self.assertEqual(
                    (
                        first.n_vertices,
                        first.n_tets,
                        first.n_triangles,
                        first.pinned_indices.size,
                        first.metadata["tip_vertex_count"],
                    ),
                    evidence["counts"],
                )
                self.assertEqual(first.metadata["source_file_sha256"], evidence["source"])
                self.assertEqual(first.metadata["source_topology_sha256"], evidence["source_topology"])
                self.assertEqual(first.metadata["topology_sha256"], evidence["topology"])
                self.assertEqual(first.metadata["material_sha256"], evidence["material"])
                self.assertEqual(first.metadata["boundary_sha256"], evidence["boundary"])
                self.assertEqual(first.manifest()["arrays"]["mass"]["sha256"], evidence["mass"])
                self.assertEqual(recording._scene_physical_sha256(first), evidence["physical"])
                self.assertEqual(first.manifest(), second.manifest())
                self.assertEqual(recording._scene_physical_sha256(first), recording._scene_physical_sha256(second))
                np.testing.assert_array_equal(recording.pin_targets_for_frame(scene_key, first, 0), first.pin_targets)

    def test_gaia_bundles_are_root_independent_and_require_v3_source_provenance(self):
        source_root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        assets = {
            "license": "LICENSE",
            "gaia-bunny-small": "Data/mesh_models/t/bunny_small.t",
            "gaia-armadilo-lowres": "Data/mesh_models/t/Armadilo_lowres.t",
        }
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_root = pathlib.Path(first_directory)
            second_root = pathlib.Path(second_directory)
            for relative in assets.values():
                for root in (first_root, second_root):
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_root / relative, target)

            for scene_key in ("gaia-bunny-small", "gaia-armadilo-lowres"):
                with self.subTest(scene=scene_key), tempfile.TemporaryDirectory() as output_directory:
                    metadata, arrays = _valid_bundle(scene_key, gaia_asset_root=first_root)
                    self.assertNotIn(str(first_root), recording._canonical_json(metadata).decode("utf-8"))

                    legacy = copy.deepcopy(metadata)
                    legacy_source = dict(legacy["generation_source"])
                    del legacy_source["manifest_sha256"]
                    legacy_source["contract"] = recording.GENERATION_SOURCE_SCHEMA_V2
                    legacy_source["files"] = {
                        name: value
                        for name, value in legacy_source["files"].items()
                        if name in recording._GENERATION_SOURCE_PATHS_V2
                    }
                    legacy["generation_source"] = _seal_manifest(legacy_source, "manifest_sha256")
                    with self.assertRaisesRegex(ValueError, "v3"):
                        _save_validated_bundle(
                            output_directory,
                            legacy,
                            arrays,
                            gaia_asset_root=first_root,
                        )

                    path = _save_validated_bundle(
                        output_directory,
                        metadata,
                        arrays,
                        gaia_asset_root=first_root,
                    )
                    bundled_root = pathlib.Path(output_directory) / recording._GAIA_BUNDLE_ROOT_RELATIVE
                    for relative in assets.values():
                        self.assertTrue((bundled_root / relative).is_file())
                    explicit_record, explicit_loaded = _load_test_bundle(path, gaia_asset_root=second_root)
                    record, loaded = _load_test_bundle(path)
                    self.assertEqual(record, explicit_record)
                    self.assertEqual(record["generation_source"]["contract"], recording.GENERATION_SOURCE_SCHEMA)
                    for name, expected in arrays.items():
                        np.testing.assert_array_equal(explicit_loaded[name], expected)
                        np.testing.assert_array_equal(loaded[name], expected)
                    with (second_root / "LICENSE").open("ab") as stream:
                        stream.write(b"tamper")
                    with self.assertRaisesRegex(ValueError, "bundle file"):
                        _load_test_bundle(path, gaia_asset_root=second_root)
                    shutil.copy2(source_root / "LICENSE", second_root / "LICENSE")
                    with (second_root / assets[scene_key]).open("ab") as stream:
                        stream.write(b"tamper")
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        _load_test_bundle(path, gaia_asset_root=second_root)
                    shutil.copy2(source_root / assets[scene_key], second_root / assets[scene_key])

    def test_gaia_bundle_copy_rejects_a_redirected_bundle_root(self):
        source_root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        with tempfile.TemporaryDirectory() as output_directory, tempfile.TemporaryDirectory() as external_directory:
            bundle_root = pathlib.Path(output_directory) / recording._GAIA_BUNDLE_ROOT_RELATIVE
            bundle_root.symlink_to(external_directory, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                recording._copy_gaia_asset_bundle(source_root, pathlib.Path(output_directory))
            with self.assertRaisesRegex(ValueError, "must not redirect"):
                recording._bundle_gaia_asset_root("gaia-bunny-small", pathlib.Path(output_directory), None)
            self.assertEqual(list(pathlib.Path(external_directory).iterdir()), [])

    def test_gaia_bundle_verification_rejects_an_intermediate_directory_symlink(self):
        source_root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        with tempfile.TemporaryDirectory() as root_directory, tempfile.TemporaryDirectory() as external_directory:
            root = pathlib.Path(root_directory)
            external = pathlib.Path(external_directory)
            shutil.copy2(source_root / "LICENSE", root / "LICENSE")
            shutil.copytree(source_root / "Data", external / "Data")
            (root / "Data").symlink_to(external / "Data", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "bundle file"):
                recording._verify_gaia_asset_bundle_files(root)


@unittest.skipUnless(
    os.environ.get("MG_VBD_TEST_CUDA") == "1" and os.environ.get("PSS_GAIA_ASSET_ROOT"),
    "set MG_VBD_TEST_CUDA=1 and PSS_GAIA_ASSET_ROOT after claiming a GPU",
)
class TestGaiaRecordingCuda(unittest.TestCase):
    def test_one_frame_integrated_recording_smoke_for_both_assets(self):
        asset_root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        for scene_key in ("gaia-bunny-small", "gaia-armadilo-lowres"):
            with self.subTest(scene=scene_key), tempfile.TemporaryDirectory() as directory:
                metadata, arrays = recording.generate_trajectory(
                    scene_key,
                    source_frames=1,
                    device="cuda:0",
                    gaia_asset_root=asset_root,
                )
                self.assertNotIn(str(asset_root), recording._canonical_json(metadata).decode("utf-8"))
                path = recording.save_content_addressed_bundle(
                    directory,
                    metadata,
                    arrays,
                    gaia_asset_root=asset_root,
                )
                record, loaded = recording.load_content_addressed_bundle(path)
                self.assertEqual(record["gaia_asset_bundle"], recording._gaia_asset_bundle_manifest(scene_key))
                self.assertTrue(np.isfinite(loaded["positions"]).all())
                self.assertTrue(np.isfinite(loaded["velocities"]).all())
                self.assertTrue(np.all(loaded["metric_determinant_min"][:, 1:] > 0.0))
                self.assertTrue(np.all(loaded["metric_inverted_tet_fraction"][:, 1:] == 0.0))
                self.assertGreaterEqual(int(loaded["mg_frame_gate_accept_count"][1]), 0)
                self.assertLessEqual(int(loaded["mg_frame_gate_accept_count"][1]), 24)


class TestGenerationSourceProvenance(unittest.TestCase):
    def test_manifest_uses_exact_ast_execution_closure(self):
        self.assertEqual(
            recording._GENERATION_SOURCE_PATHS,
            (
                "research/principal_stretch/__init__.py",
                "research/principal_stretch/record_mg_vbd_comparison.py",
                "research/principal_stretch/mg_vbd_rollout.py",
                "research/principal_stretch/captured_graph_vbd.py",
                "research/principal_stretch/correction_graph_vbd.py",
                "research/principal_stretch/correction_mg_vbd.py",
                "research/principal_stretch/correction_multigrid.py",
                "research/principal_stretch/solver_benchmark.py",
                "research/principal_stretch/solver_scenes.py",
                "research/principal_stretch/newton_baseline.py",
                "research/principal_stretch/potentials.py",
                "research/principal_stretch/captured_mg_vbd.py",
                "research/principal_stretch/captured_vbd_baseline.py",
                "research/principal_stretch/correction_gpu.py",
                "research/principal_stretch/correction_gpu_warp.py",
                "research/principal_stretch/correction_multigrid_warp.py",
                "research/principal_stretch/correction_multigrid_warp_scalar_fused.py",
                "research/principal_stretch/polar.py",
                "research/principal_stretch/sparse_newton_reference.py",
                "research/principal_stretch/torch_solver.py",
                "research/principal_stretch/gaia_assets.py",
            ),
        )

    def test_published_v2_generation_source_closure_remains_valid(self):
        current = _fake_generation_source()
        legacy = copy.deepcopy(current)
        del legacy["manifest_sha256"]
        legacy["contract"] = recording.GENERATION_SOURCE_SCHEMA_V2
        legacy["files"] = {
            name: value for name, value in legacy["files"].items() if name in recording._GENERATION_SOURCE_PATHS_V2
        }
        legacy = _seal_manifest(legacy, "manifest_sha256")
        self.assertEqual(recording._validate_generation_source_manifest(legacy), legacy)

    def test_historical_recorded_commit_and_raw_blobs_verify_after_head_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _generation_repository(directory)
            manifest = recording._generation_source_manifest(repository)
            original_revision = manifest["git_revision"]
            changed = repository / recording._GENERATION_SOURCE_PATHS[-1]
            changed.write_text("# later committed source\n", encoding="utf-8")
            _git(repository, "add", "--", recording._GENERATION_SOURCE_PATHS[-1])
            _git(repository, "commit", "--quiet", "-m", "Advance repository head")
            self.assertNotEqual(_git(repository, "rev-parse", "HEAD"), original_revision)
            recording._verify_generation_source_git_objects(manifest, repository=repository)

    def test_rehashed_git_object_forgeries_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _generation_repository(directory)
            manifest = recording._generation_source_manifest(repository)
            first_path = recording._GENERATION_SOURCE_PATHS[0]
            mutations = {
                "nonexistent-revision": lambda item: item.__setitem__("git_revision", "0" * 40),
                "invented-tree": lambda item: item.__setitem__("git_tree_oid", "0" * 40),
                "invented-blob": lambda item: item["files"][first_path].__setitem__("git_blob_oid", "0" * 40),
                "raw-blob-sha": lambda item: item["files"][first_path].__setitem__("sha256", "0" * 64),
            }
            for name, mutate in mutations.items():
                with self.subTest(mutation=name):
                    changed = copy.deepcopy(manifest)
                    del changed["manifest_sha256"]
                    mutate(changed)
                    changed = _seal_manifest(changed, "manifest_sha256")
                    with self.assertRaises(ValueError):
                        recording._verify_generation_source_git_objects(changed, repository=repository)

    def test_generation_rejects_untracked_and_dirty_omitted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _generation_repository(directory)
            recording._generation_source_manifest(repository)
            transitive_dependency = repository / "research/principal_stretch/torch_solver.py"
            transitive_dependency.write_text("# dirty transitive dependency\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completely clean"):
                recording._generation_source_manifest(repository)
            _git(repository, "add", "--", "research/principal_stretch/torch_solver.py")
            _git(repository, "commit", "--quiet", "-m", "Commit changed transitive dependency")
            omitted = repository / "outside-execution-closure.txt"
            omitted.write_text("untracked\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completely clean"):
                recording._generation_source_manifest(repository)
            _git(repository, "add", "--", omitted.name)
            _git(repository, "commit", "--quiet", "-m", "Commit omitted file")
            omitted.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completely clean"):
                recording._generation_source_manifest(repository)


class TestContentAddressedBundle(unittest.TestCase):
    def test_published_v1_bundle_with_v2_source_manifest_still_loads(self):
        metadata, expected_arrays = _valid_bundle()
        metadata["schema"] = recording.SCHEMA_V1
        del metadata["gaia_asset_bundle"]
        del metadata["execution_authentication"]
        scene_manifest = copy.deepcopy(metadata["scene_manifest"])
        del scene_manifest["scene_sha256"]
        scene_manifest["metadata"]["newton_revision"] = metadata["git_revision"]
        scene_manifest["metadata"]["dirty_tree_sha256"] = None
        metadata["scene_manifest"] = _seal_manifest(scene_manifest, "scene_sha256")
        source = dict(metadata["generation_source"])
        del source["manifest_sha256"]
        source["contract"] = recording.GENERATION_SOURCE_SCHEMA_V2
        source["files"] = {
            name: value for name, value in source["files"].items() if name in recording._GENERATION_SOURCE_PATHS_V2
        }
        metadata["generation_source"] = _seal_manifest(source, "manifest_sha256")
        with tempfile.TemporaryDirectory() as directory:
            path = _write_unvalidated_bundle(pathlib.Path(directory), metadata, expected_arrays)
            record, arrays = _load_test_bundle(path)
        self.assertEqual(record["schema"], recording.SCHEMA_V1)
        for name, expected in expected_arrays.items():
            np.testing.assert_array_equal(arrays[name], expected)

    def test_current_v2_bundle_rejects_legacy_generation_source_closure(self):
        metadata, arrays = _valid_bundle()
        source = dict(metadata["generation_source"])
        del source["manifest_sha256"]
        source["contract"] = recording.GENERATION_SOURCE_SCHEMA_V2
        source["files"] = {
            name: value for name, value in source["files"].items() if name in recording._GENERATION_SOURCE_PATHS_V2
        }
        metadata["generation_source"] = _seal_manifest(source, "manifest_sha256")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "v3"):
                _save_validated_bundle(directory, metadata, arrays)

    def test_round_trip_verifies_semantic_json_and_npz_bytes(self):
        metadata, expected_arrays = _valid_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = _save_validated_bundle(directory, metadata, expected_arrays)
            with mock.patch.object(recording, "_verify_generation_source_git_objects") as verifier:
                record, arrays = recording.load_content_addressed_bundle(path)
            verifier.assert_called_once_with(metadata["generation_source"])
            self.assertTrue(path.name.endswith(f"-{record['record_sha256']}.json"))
            self.assertTrue(record["npz_filename"].endswith(f"-{record['npz_file_sha256']}.npz"))
            for name, expected in expected_arrays.items():
                np.testing.assert_array_equal(arrays[name], expected)

    def test_json_tamper_fails_closed(self):
        metadata, arrays = _valid_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = _save_validated_bundle(directory, metadata, arrays)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["scene_display_name"] = "tampered"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantic digest"):
                _load_test_bundle(path)

    def test_sealed_v2_bundle_accepts_only_clean_historical_scene_provenance(self):
        metadata, expected_arrays = _valid_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = _save_validated_bundle(directory, metadata, expected_arrays)
            record, arrays = _load_test_bundle(path)
        self.assertEqual(record["schema"], recording.SCHEMA)
        for name, expected in expected_arrays.items():
            np.testing.assert_array_equal(arrays[name], expected)

        for mutation in ("schema", "density", "boolean-type"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(metadata)
                changed_manifest = copy.deepcopy(changed["scene_manifest"])
                del changed_manifest["scene_sha256"]
                if mutation == "schema":
                    changed_manifest["schema_version"] = "fabricated-scene-schema"
                elif mutation == "density":
                    changed_manifest["metadata"]["density_kg_m3"] = 1.0
                else:
                    changed_manifest["metadata"]["contact"] = 0
                changed["scene_manifest"] = _seal_manifest(changed_manifest, "scene_sha256")
                path = _write_unvalidated_bundle(pathlib.Path(directory), changed, expected_arrays)
                with self.assertRaisesRegex(ValueError, "historical scene"):
                    _load_test_bundle(path)

    def test_npz_tamper_fails_closed(self):
        metadata, arrays = _valid_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = _save_validated_bundle(directory, metadata, arrays)
            record = json.loads(path.read_text(encoding="utf-8"))
            npz_path = pathlib.Path(directory) / record["npz_filename"]
            with npz_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "NPZ bytes"):
                _load_test_bundle(path)

    def test_content_addressed_files_must_not_redirect_through_symlinks(self):
        metadata, arrays = _valid_bundle()
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external_directory:
            path = _save_validated_bundle(directory, metadata, arrays)
            record = json.loads(path.read_text(encoding="utf-8"))
            npz_path = pathlib.Path(directory) / record["npz_filename"]
            external_npz = pathlib.Path(external_directory) / npz_path.name
            npz_path.replace(external_npz)
            npz_path.symlink_to(external_npz)
            with self.assertRaisesRegex(ValueError, "NPZ must not be a symlink"):
                _load_test_bundle(path)
            with self.assertRaisesRegex(RuntimeError, "NPZ must not be a symlink"):
                _save_validated_bundle(directory, metadata, arrays)

    def test_save_rejects_semantic_mutations(self):
        def delete_position(metadata, arrays):
            del arrays["positions"]

        def wrong_dtype(metadata, arrays):
            arrays["velocities"] = arrays["velocities"].astype(np.float32)

        def nonfinite_position(metadata, arrays):
            arrays["positions"][0, 1, 0, 0] = np.nan

        def wrong_timeline(metadata, arrays):
            arrays["time_seconds"][1] += 1.0e-6

        def wrong_pin(metadata, arrays):
            scene = recording.build_recording_scene("twist")
            arrays["positions"][0, 1, scene.pinned_indices[0], 0] += 1.0e-4

        def wrong_initial_state(metadata, arrays):
            arrays["velocities"][0, 0, 0, 0] += 1.0

        def wrong_gate(metadata, arrays):
            arrays["mg_last_gate_reason_code"][1, 0] = 0

        def wrong_method_claim_type(metadata, arrays):
            metadata["methods"][0]["ground_truth_claim"] = 0

        def wrong_reference_claim_type(metadata, arrays):
            metadata["reference_policy"]["ground_truth_claim"] = 0

        def wrong_execution_authentication_type(metadata, arrays):
            metadata["execution_authentication"]["offline_cryptographic_attestation"] = 0

        def wrong_scene_manifest(metadata, arrays):
            metadata["scene_manifest"]["n_vertices"] += 1

        def wrong_camera(metadata, arrays):
            metadata["camera"]["position"][0] += 1.0

        def wrong_metric(metadata, arrays):
            arrays["metric_objective"][1, 1] += 1.0

        def odd_dimensions(metadata, arrays):
            metadata["camera"]["panel_width"] = 639

        mutations = (
            delete_position,
            wrong_dtype,
            nonfinite_position,
            wrong_timeline,
            wrong_pin,
            wrong_initial_state,
            wrong_gate,
            wrong_method_claim_type,
            wrong_reference_claim_type,
            wrong_execution_authentication_type,
            wrong_scene_manifest,
            wrong_camera,
            wrong_metric,
            odd_dimensions,
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate.__name__), tempfile.TemporaryDirectory() as directory:
                metadata, arrays = _valid_bundle()
                mutate(metadata, arrays)
                with self.assertRaises(ValueError):
                    _save_validated_bundle(directory, metadata, arrays)

    def test_save_rejects_a_correctly_scored_inverted_trajectory(self):
        metadata, arrays = _valid_bundle()
        scene = recording.build_recording_scene("twist")
        free = {int(index) for index in scene.free_indices}
        tet = next(indices for indices in scene.tet_indices if all(int(index) in free for index in indices))
        first, second = (int(tet[0]), int(tet[1]))
        arrays["positions"][:, 1, [first, second]] = arrays["positions"][:, 1, [second, first]]
        for method in range(len(recording.METHOD_IDS)):
            objective_scene = dataclasses.replace(
                scene,
                x_current=arrays["objective_input_positions"][method, 1],
                velocity=arrays["objective_input_velocities"][method, 1],
                pin_targets=arrays["pin_targets"][1],
            )
            metrics = recording.evaluate_common_state(
                recording.build_common_problem(objective_scene),
                arrays["positions"][method, 1],
                reference_positions=arrays["positions"][0, 1],
            )
            self.assertLess(metrics.determinant_min, 0.0)
            recording._store_metrics(arrays, method, 1, metrics)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-positive|inverted"):
                _save_validated_bundle(directory, metadata, arrays)

    def test_save_rejects_an_inverted_stored_objective_input_state(self):
        metadata, arrays = _valid_bundle()
        scene = recording.build_recording_scene("twist")
        free = {int(index) for index in scene.free_indices}
        tet = next(indices for indices in scene.tet_indices if all(int(index) in free for index in indices))
        first, second = (int(tet[0]), int(tet[1]))
        arrays["objective_input_positions"][:, 1, [first, second]] = arrays["objective_input_positions"][
            :, 1, [second, first]
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "objective-input trajectory"):
                _save_validated_bundle(directory, metadata, arrays)

    def test_sealing_requires_the_exact_one_shot_process_generation_receipt(self):
        metadata, arrays = _valid_bundle()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                recording,
                "_generation_source_manifest",
                return_value=metadata["generation_source"],
            ):
                with self.assertRaisesRegex(RuntimeError, "not returned by this process"):
                    recording.save_content_addressed_bundle(directory, metadata, arrays)

                recording._issue_generated_trajectory(metadata, arrays)
                with self.assertRaisesRegex(RuntimeError, "not returned by this process"):
                    recording.save_content_addressed_bundle(directory, copy.deepcopy(metadata), arrays)

                arrays["solve_seconds"][0, 1] += 0.125
                with self.assertRaisesRegex(RuntimeError, "changed after"):
                    recording.save_content_addressed_bundle(directory, metadata, arrays)
                arrays["solve_seconds"][0, 1] -= 0.125

                path = recording.save_content_addressed_bundle(directory, metadata, arrays)
                self.assertTrue(path.is_file())
                with self.assertRaisesRegex(RuntimeError, "not returned by this process"):
                    recording.save_content_addressed_bundle(directory, metadata, arrays)

    def test_sealing_uses_one_frozen_metadata_snapshot_across_receipt_consumption(self):
        metadata, arrays = _valid_bundle()
        recording._issue_generated_trajectory(metadata, arrays)
        consume = recording._consume_generated_trajectory

        def consume_then_mutate(*args):
            consume(*args)
            metadata["setup_seconds_diagnostic"]["reference_public_vbd"] = 123.0

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    recording,
                    "_generation_source_manifest",
                    return_value=metadata["generation_source"],
                ),
                mock.patch.object(recording, "_consume_generated_trajectory", side_effect=consume_then_mutate),
            ):
                path = recording.save_content_addressed_bundle(directory, metadata, arrays)
            record, _loaded = _load_test_bundle(path)
        self.assertEqual(record["setup_seconds_diagnostic"]["reference_public_vbd"], 0.0)
        self.assertEqual(metadata["setup_seconds_diagnostic"]["reference_public_vbd"], 123.0)

    def test_sealing_uses_independent_array_snapshots_across_receipt_consumption(self):
        metadata, arrays = _valid_bundle()
        recording._issue_generated_trajectory(metadata, arrays)
        consume = recording._consume_generated_trajectory

        def consume_then_mutate(*args):
            consume(*args)
            snapshots = args[3]
            self.assertTrue(all(not np.shares_memory(arrays[name], snapshots[name]) for name in arrays))
            arrays["solve_seconds"][0, 1] = 456.0

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    recording,
                    "_generation_source_manifest",
                    return_value=metadata["generation_source"],
                ),
                mock.patch.object(recording, "_consume_generated_trajectory", side_effect=consume_then_mutate),
            ):
                path = recording.save_content_addressed_bundle(directory, metadata, arrays)
            _record, loaded = _load_test_bundle(path)
        self.assertEqual(loaded["solve_seconds"][0, 1], 0.0)
        self.assertEqual(arrays["solve_seconds"][0, 1], 456.0)

    def test_sealing_io_failure_releases_the_receipt_for_an_exact_retry(self):
        metadata, arrays = _valid_bundle()
        recording._issue_generated_trajectory(metadata, arrays)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                recording,
                "_generation_source_manifest",
                return_value=metadata["generation_source"],
            ):
                with (
                    mock.patch.object(recording.np, "savez_compressed", side_effect=OSError("forced I/O failure")),
                    self.assertRaisesRegex(OSError, "forced I/O failure"),
                ):
                    recording.save_content_addressed_bundle(directory, metadata, arrays)
                path = recording.save_content_addressed_bundle(directory, metadata, arrays)
                self.assertTrue(path.is_file())

    def test_sealing_json_replace_failure_cleans_temp_and_allows_exact_retry(self):
        metadata, arrays = _valid_bundle()
        recording._issue_generated_trajectory(metadata, arrays)
        replace = recording.os.replace

        def fail_json_replace(source, destination):
            if pathlib.Path(source).name.startswith(".record-"):
                raise OSError("forced JSON replace failure")
            return replace(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                recording,
                "_generation_source_manifest",
                return_value=metadata["generation_source"],
            ):
                with (
                    mock.patch.object(recording.os, "replace", side_effect=fail_json_replace),
                    self.assertRaisesRegex(OSError, "forced JSON replace failure"),
                ):
                    recording.save_content_addressed_bundle(directory, metadata, arrays)
                self.assertFalse(any(pathlib.Path(directory).glob(".record-*.json")))
                path = recording.save_content_addressed_bundle(directory, metadata, arrays)
                self.assertTrue(path.is_file())

    def test_load_rejects_fully_rehashed_semantic_mutation(self):
        metadata, arrays = _valid_bundle()
        metadata["camera"]["target"][2] += 0.25
        with tempfile.TemporaryDirectory() as directory:
            path = _write_unvalidated_bundle(pathlib.Path(directory), metadata, arrays)
            with self.assertRaisesRegex(ValueError, "fixed camera"):
                _load_test_bundle(path)

    def test_load_rejects_fully_rehashed_manifest_and_filename_mutations(self):
        for mutation in ("array-manifest-extra", "npz-path"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                metadata, arrays = _valid_bundle()
                original = _write_unvalidated_bundle(pathlib.Path(directory), metadata, arrays)
                record = json.loads(original.read_text(encoding="utf-8"))
                del record["record_sha256"]
                if mutation == "array-manifest-extra":
                    record["arrays"]["positions"]["unexpected"] = True
                else:
                    record["npz_filename"] = f"nested/{record['npz_filename']}"
                record_sha256 = hashlib.sha256(recording._canonical_json(record)).hexdigest()
                record["record_sha256"] = record_sha256
                mutated = pathlib.Path(directory) / f"twist-{record_sha256}.json"
                mutated.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _load_test_bundle(mutated)

    def test_load_rejects_fully_rehashed_json_numeric_type_aliases(self):
        def array_shape_float(record):
            record["arrays"]["positions"]["shape"][0] = 3.0

        def source_rate_float(record):
            record["simulation"]["source_frame_rate_hz"] = 60.0

        def stored_frames_float(record):
            record["simulation"]["stored_frames_including_initial"] = 2.0

        def substeps_float(record):
            record["simulation"]["substeps_per_source_frame"] = float(record["simulation"]["substeps_per_source_frame"])

        def camera_number_type(record):
            record["camera"]["fov_degrees"] = 32

        for mutate in (array_shape_float, source_rate_float, stored_frames_float, substeps_float, camera_number_type):
            with self.subTest(mutation=mutate.__name__), tempfile.TemporaryDirectory() as directory:
                metadata, arrays = _valid_bundle()
                original = _write_unvalidated_bundle(pathlib.Path(directory), metadata, arrays)
                record = json.loads(original.read_text(encoding="utf-8"))
                del record["record_sha256"]
                mutate(record)
                record_sha256 = hashlib.sha256(recording._canonical_json(record)).hexdigest()
                record["record_sha256"] = record_sha256
                mutated = pathlib.Path(directory) / f"twist-{record_sha256}.json"
                mutated.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _load_test_bundle(mutated)

    def test_generation_source_manifest_rejects_dirty_or_incomplete_repository(self):
        manifest = _fake_generation_source()
        recording._validate_generation_source_manifest(manifest)
        for mutate in ("dirty", "missing"):
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(manifest)
                del changed["manifest_sha256"]
                if mutate == "dirty":
                    changed["repository_clean"] = False
                else:
                    del changed["files"][recording._GENERATION_SOURCE_PATHS[0]]
                changed = _seal_manifest(changed, "manifest_sha256")
                with self.assertRaises(ValueError):
                    recording._validate_generation_source_manifest(changed)

        metadata, arrays = _valid_bundle()
        changed = copy.deepcopy(metadata["generation_source"])
        del changed["manifest_sha256"]
        changed["newton_version"] = "changed-after-generation"
        changed = _seal_manifest(changed, "manifest_sha256")
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(recording, "_generation_source_manifest", return_value=changed),
                mock.patch.object(recording, "_consume_generated_trajectory"),
            ):
                with self.assertRaisesRegex(ValueError, "changed between simulation and sealing"):
                    recording.save_content_addressed_bundle(directory, metadata, arrays)


class TestAuthenticatedStaticReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scene = recording.build_recording_scene("refinement-medium")
        cls.endpoint_positions = run_newton(scene, warmup=False, repeats=1).result.x.detach().numpy()

    @staticmethod
    def _source_record(
        role: str,
        scene_sha256: str,
        objective_sha256: str,
        position_sha256: str,
        metrics,
        residual_scale: float,
    ) -> dict[str, object]:
        method, contract = recording._STATIC_REFERENCE_METHODS[role]
        config = {
            "max_iterations": 50,
            "gradient_absolute_tolerance": 1.0e-10,
            "gradient_relative_tolerance": 1.0e-10,
            "step_relative_tolerance": 1.0e-14,
            "armijo": 1.0e-4,
            "backtrack": 0.5,
            "max_line_search_steps": 30,
            "minimum_eigenvalue_relative": 1.0e-9,
            "regularization_growth": 10.0,
            "max_regularization_attempts": 12,
        }
        source: dict[str, object] = {
            "method": method,
            "contract": contract,
            "config": config,
            "accepted": True,
            "failures": [],
            "native_converged": True,
            "native_reason": "gradient",
            "accepted_iterations": 0 if role == "dense" else 1,
            "final_objective": metrics.objective,
            "final_gradient_norm": metrics.gradient_norm,
            "final_relative_residual": metrics.relative_residual,
            "verification_converged": True,
            "verification_reason": "gradient",
            "verification_displacement_relative": 0.0,
            "alternate_start_converged": True,
            "alternate_start_reason": "gradient",
            "alternate_start_displacement_relative": 0.0,
            "scene_sha256": scene_sha256,
            "objective_instance_sha256": objective_sha256,
            "position_sha256": position_sha256,
        }
        if role == "dense":
            return source

        accepted = dict.fromkeys(recording._SPARSE_TRACE_NAMES)
        accepted.update(
            {
                "iteration": 0,
                "objective": metrics.objective + 1.0,
                "gradient_norm": residual_scale,
                "relative_residual": 1.0,
                "minimum_determinant": metrics.determinant_min,
                "hessian_nnz": 1,
                "minimum_eigenvalue": 1.0,
                "eigenpair_residual": 0.0,
                "diagonal_scale": 1.0,
                "ritz_regularization": 0.0,
                "gershgorin_rescue_used": False,
                "regularization": 0.0,
                "last_attempted_regularization": 0.0,
                "factor_nnz": 1,
                "factorization_attempts": 1,
                "factor_certificate_attempts": 1,
                "linear_solve_attempts": 1,
                "linear_refinement_steps": 0,
                "line_search_trials": 1,
                "factor_permutations_match": True,
                "factor_l_unit_diagonal_error": 0.0,
                "factor_minimum_diagonal": 1.0,
                "factor_maximum_diagonal_magnitude": 1.0,
                "factor_minimum_diagonal_relative": 1.0,
                "factor_relation_relative_residual": 0.0,
                "factorization_relative_residual": 0.0,
                "factor_certificate_passed": True,
                "linear_relative_residual": 0.0,
                "directional_derivative": -1.0,
                "accepted_step_norm": 1.0e-6,
                "accepted_step_size": 1.0,
            }
        )
        terminal = dict.fromkeys(recording._SPARSE_TRACE_NAMES)
        terminal.update(
            {
                "iteration": 1,
                "objective": metrics.objective,
                "gradient_norm": metrics.gradient_norm,
                "relative_residual": metrics.relative_residual,
                "minimum_determinant": metrics.determinant_min,
                "hessian_nnz": 1,
                "gershgorin_rescue_used": False,
                "factorization_attempts": 0,
                "factor_certificate_attempts": 0,
                "linear_solve_attempts": 0,
                "linear_refinement_steps": 0,
                "line_search_trials": 0,
            }
        )
        result = {
            "contract": "exact-sparse-cpu-newton-float64-v2",
            "hessian_contract": "exact-stable-nh-element-hessian-csr-v1",
            "linear_solver": "scipy-superlu-mmd-at-plus-a-symmetric-ldlt-certified-v2",
            "eigen_policy": "arpack-smallest-algebraic-ritz-heuristic-gershgorin-rescue-v2",
            "factor_certificate": "superlu-symmetric-ldlt-numerical-certificate-v2",
            "factor_equilibration": False,
            "factor_unit_diagonal_limit": 1.4210854715202004e-14,
            "factor_pivot_relative_margin": 1.1368683772161603e-13,
            "factor_relation_relative_limit": 5.0e-12,
            "factorization_relative_residual_limit": 5.0e-12,
            "linear_residual_limit": 5.0e-13,
            "maximum_refinement_steps": 4,
            "positions_sha256": position_sha256,
            "converged": True,
            "reason": "gradient",
            "accepted_iterations": 1,
            "residual_scale": residual_scale,
            "scipy_version": "unit-test-scipy",
            "final_objective": metrics.objective,
            "final_gradient_norm": metrics.gradient_norm,
            "final_relative_residual": metrics.relative_residual,
            "work": {
                "objective_evaluations": 3,
                "gradient_evaluations": 2,
                "hessian_evaluations": 2,
                "eigenvalue_evaluations": 1,
                "factorization_attempts": 1,
                "factor_certificate_attempts": 1,
                "linear_solve_attempts": 1,
                "line_search_trials": 1,
            },
            "trace": [accepted, terminal],
        }
        native_sha256 = hashlib.sha256(recording._canonical_json(result)).hexdigest()
        source.update(
            {
                "alternate_start_gradient_norm": metrics.gradient_norm,
                "alternate_start_relative_residual": metrics.relative_residual,
                "repeat_count": 1,
                "repeat_deterministic_sha256": [native_sha256],
                "native_result": copy.deepcopy(result),
                "verification_result": copy.deepcopy(result),
                "alternate_start_result": copy.deepcopy(result),
            }
        )
        return source

    def _write_bundle(self, directory: str) -> tuple[pathlib.Path, pathlib.Path, str, str]:
        root = pathlib.Path(directory)
        scene = recording.build_recording_scene("refinement-medium")
        dense = np.array(self.endpoint_positions, dtype=np.float64, copy=True)
        sparse = dense.copy()
        npz_path = root / "reference.npz"
        np.savez_compressed(npz_path, dense_positions=dense, sparse_positions=sparse)
        problem = recording.build_common_problem(scene)
        scene_sha256 = scene.manifest()["scene_sha256"]
        objective_sha256 = recording.common_objective_manifest(scene, problem)["objective_instance_sha256"]
        entries = {}
        arrays = {}
        for role, array in (("dense", dense), ("sparse", sparse)):
            position_sha256 = recording.array_sha256(array)
            metrics = recording.evaluate_common_state(
                problem,
                array,
                reference_positions=array,
            )
            source_record = self._source_record(
                role,
                scene_sha256,
                objective_sha256,
                position_sha256,
                metrics,
                problem.residual_scale,
            )
            entries[role] = {
                "method": recording._STATIC_REFERENCE_METHODS[role][0],
                "source_record": source_record,
                "source_record_sha256": hashlib.sha256(recording._canonical_json(source_record)).hexdigest(),
                "independent_metrics": metrics.as_dict(),
            }
            arrays[f"{role}_positions"] = {
                "dtype": "float64",
                "shape": [525, 3],
                "array_sha256": position_sha256,
            }
        record = {
            "schema": recording.STATIC_REFERENCE_SCHEMA,
            "scene_name": "pr2901-refinement-medium-common-step",
            "scene_sha256": scene_sha256,
            "scene_physical_sha256": recording._scene_physical_sha256(scene),
            "objective_instance_sha256": objective_sha256,
            "git_revision": recording._git_revision(),
            "vertices": scene.n_vertices,
            "tets": scene.n_tets,
            "free_dofs": int(scene.free_indices.size * 3),
            "npz_path": str(npz_path),
            "arrays": arrays,
            "dense": entries["dense"],
            "sparse": entries["sparse"],
            "comparison": recording._static_comparison(
                scene,
                dense,
                sparse,
                recording.evaluate_common_state(
                    recording.build_common_problem(scene), dense, reference_positions=dense
                ),
                recording.evaluate_common_state(
                    recording.build_common_problem(scene), sparse, reference_positions=sparse
                ),
            ),
        }
        json_path = root / "reference.json"
        json_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        return (
            json_path,
            npz_path,
            recording._file_sha256(json_path),
            recording._file_sha256(npz_path),
        )

    def test_both_files_and_both_source_records_are_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path, json_sha256, npz_sha256 = self._write_bundle(directory)
            record, arrays = recording.load_authenticated_medium_reference(
                json_path,
                expected_json_sha256=json_sha256,
                expected_npz_sha256=npz_sha256,
                npz_path=npz_path,
            )
            self.assertEqual(record["schema"], recording.STATIC_REFERENCE_SCHEMA)
            self.assertEqual(set(arrays), {"dense_positions", "sparse_positions"})

    def test_endpoint_gate_rejects_inversion_nonpositive_determinant_and_pin_error(self):
        scene = recording.build_recording_scene("refinement-medium")
        metrics = recording.evaluate_common_state(
            recording.build_common_problem(scene),
            self.endpoint_positions,
            reference_positions=self.endpoint_positions,
        )
        recording._validate_static_endpoint_metrics(metrics, role="valid")
        mutations = (
            dataclasses.replace(metrics, inverted_tet_fraction=1.0 / scene.n_tets),
            dataclasses.replace(metrics, determinant_min=0.0),
            dataclasses.replace(metrics, max_pin_error_m=1.0e-12),
        )
        for changed in mutations:
            with self.subTest(metrics=changed), self.assertRaises(ValueError):
                recording._validate_static_endpoint_metrics(changed, role="mutated")

    def test_wrong_method_contract_fails_even_with_updated_file_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path, _json_sha256, npz_sha256 = self._write_bundle(directory)
            record = json.loads(json_path.read_text(encoding="utf-8"))
            record["sparse"]["method"] = "dense-cpu-newton-float64"
            json_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "method"):
                recording.load_authenticated_medium_reference(
                    json_path,
                    expected_json_sha256=recording._file_sha256(json_path),
                    expected_npz_sha256=npz_sha256,
                    npz_path=npz_path,
                )

    def test_rehashed_dense_iteration_type_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path, _json_sha256, npz_sha256 = self._write_bundle(directory)
            record = json.loads(json_path.read_text(encoding="utf-8"))
            dense = record["dense"]
            dense["source_record"]["accepted_iterations"] = "forged"
            dense["source_record_sha256"] = hashlib.sha256(
                recording._canonical_json(dense["source_record"])
            ).hexdigest()
            json_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted iteration"):
                recording.load_authenticated_medium_reference(
                    json_path,
                    expected_json_sha256=recording._file_sha256(json_path),
                    expected_npz_sha256=npz_sha256,
                    npz_path=npz_path,
                )

    def test_rehashed_sparse_v2_semantic_mutations_fail_canonical_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path, _json_sha256, npz_sha256 = self._write_bundle(directory)
            mutations = (
                (
                    "v1",
                    lambda source: source.__setitem__("contract", "fresh-sparse-exact-newton-accepted-reference-v1"),
                    False,
                ),
                (
                    "certificate",
                    lambda source: source["native_result"]["trace"][0].__setitem__("factor_certificate_passed", False),
                    True,
                ),
                (
                    "linear-residual",
                    lambda source: source["native_result"]["trace"][0].__setitem__("linear_relative_residual", 1.0),
                    True,
                ),
                (
                    "gershgorin",
                    lambda source: source["native_result"]["trace"][0].__setitem__("gershgorin_lower_bound", 0.0),
                    True,
                ),
                (
                    "last-shift",
                    lambda source: source["native_result"]["trace"][0].__setitem__(
                        "last_attempted_regularization", 1.0
                    ),
                    True,
                ),
                (
                    "work",
                    lambda source: source["native_result"]["work"].__setitem__("objective_evaluations", 4),
                    True,
                ),
                (
                    "repeat",
                    lambda source: source["repeat_deterministic_sha256"].__setitem__(0, "0" * 64),
                    False,
                ),
                (
                    "config-extra",
                    lambda source: source["config"].__setitem__("unexpected", True),
                    False,
                ),
                (
                    "nested-extra",
                    lambda source: source["native_result"].__setitem__("unexpected", True),
                    True,
                ),
            )
            for name, mutate, rehash_repeat in mutations:
                with self.subTest(mutation=name):
                    record = json.loads(json_path.read_text(encoding="utf-8"))
                    sparse = record["sparse"]
                    mutate(sparse["source_record"])
                    if rehash_repeat:
                        native_sha256 = hashlib.sha256(
                            recording._canonical_json(sparse["source_record"]["native_result"])
                        ).hexdigest()
                        sparse["source_record"]["repeat_deterministic_sha256"] = [native_sha256]
                    sparse["source_record_sha256"] = hashlib.sha256(
                        recording._canonical_json(sparse["source_record"])
                    ).hexdigest()
                    mutated = pathlib.Path(directory) / f"{name}.json"
                    mutated.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        recording.load_authenticated_medium_reference(
                            mutated,
                            expected_json_sha256=recording._file_sha256(mutated),
                            expected_npz_sha256=npz_sha256,
                            npz_path=npz_path,
                        )

    def test_static_evidence_recomputes_comparison_and_checks_physical_scene(self):
        scene = recording.build_recording_scene("refinement-medium")
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path, json_sha256, npz_sha256 = self._write_bundle(directory)
            evidence = recording.static_medium_reference_evidence(
                scene,
                json_path,
                expected_json_sha256=json_sha256,
                expected_npz_sha256=npz_sha256,
                npz_path=npz_path,
            )
            self.assertEqual(evidence["comparison"], dict.fromkeys(recording._STATIC_COMPARISON_NAMES, 0.0))
            recording._validate_compact_static_reference(evidence, scene, require_current=True)
            stale_compact = copy.deepcopy(evidence)
            stale_compact["comparison"]["objective_delta_sparse_minus_dense"] = 1.0
            with self.assertRaisesRegex(ValueError, "comparison"):
                recording._validate_compact_static_reference(stale_compact, scene, require_current=True)

            record = json.loads(json_path.read_text(encoding="utf-8"))
            record["comparison"]["free_rms_m"] = 1.0
            json_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comparison"):
                recording.static_medium_reference_evidence(
                    scene,
                    json_path,
                    expected_json_sha256=recording._file_sha256(json_path),
                    expected_npz_sha256=npz_sha256,
                    npz_path=npz_path,
                )

            record["comparison"]["free_rms_m"] = 0.0
            record["scene_physical_sha256"] = "0" * 64
            json_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "physical scene"):
                recording.static_medium_reference_evidence(
                    scene,
                    json_path,
                    expected_json_sha256=recording._file_sha256(json_path),
                    expected_npz_sha256=npz_sha256,
                    npz_path=npz_path,
                )

    def test_rehashed_inverted_npz_fails_integrated_endpoint_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path, _json_sha256, _npz_sha256 = self._write_bundle(directory)
            record = json.loads(json_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as archive:
                dense = np.array(archive["dense_positions"], copy=True)
                sparse = np.array(archive["sparse_positions"], copy=True)
            scene = recording.build_recording_scene("refinement-medium")
            free = {int(index) for index in scene.free_indices}
            tet = next(indices for indices in scene.tet_indices if all(int(index) in free for index in indices))
            first, second = (int(tet[0]), int(tet[1]))
            dense[[first, second]] = dense[[second, first]]
            problem = recording.build_common_problem(scene)
            metrics = recording.evaluate_common_state(problem, dense, reference_positions=dense)
            self.assertLess(metrics.determinant_min, 0.0)
            self.assertGreater(metrics.inverted_tet_fraction, 0.0)

            mutated_npz = pathlib.Path(directory) / "inverted.npz"
            np.savez_compressed(mutated_npz, dense_positions=dense, sparse_positions=sparse)
            dense_sha256 = recording.array_sha256(dense)
            record["arrays"]["dense_positions"]["array_sha256"] = dense_sha256
            record["dense"]["independent_metrics"] = metrics.as_dict()
            dense_source = record["dense"]["source_record"]
            dense_source.update(
                {
                    "position_sha256": dense_sha256,
                    "final_objective": metrics.objective,
                    "final_gradient_norm": metrics.gradient_norm,
                    "final_relative_residual": metrics.relative_residual,
                }
            )
            record["dense"]["source_record_sha256"] = hashlib.sha256(
                recording._canonical_json(dense_source)
            ).hexdigest()
            json_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inverted|non-positive"):
                recording.load_authenticated_medium_reference(
                    json_path,
                    expected_json_sha256=recording._file_sha256(json_path),
                    expected_npz_sha256=recording._file_sha256(mutated_npz),
                    npz_path=mutated_npz,
                )

    def test_rehashed_outer_bundle_rejects_compact_endpoint_gate_mutations(self):
        scene = recording.build_recording_scene("refinement-medium")
        with tempfile.TemporaryDirectory() as reference_directory:
            json_path, npz_path, json_sha256, npz_sha256 = self._write_bundle(reference_directory)
            evidence = recording.static_medium_reference_evidence(
                scene,
                json_path,
                expected_json_sha256=json_sha256,
                expected_npz_sha256=npz_sha256,
                npz_path=npz_path,
            )
        for field, value in (
            ("determinant_min", 0.0),
            ("inverted_tet_fraction", 1.0 / scene.n_tets),
            ("max_pin_error_m", 1.0e-12),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                metadata, arrays = _valid_bundle("refinement-medium")
                compact = copy.deepcopy(evidence)
                historical_scene, _current_scene, historical_objective = (
                    recording._historical_static_reference_identities(scene, metadata["git_revision"])
                )
                compact["source_scene_sha256"] = historical_scene
                compact["current_scene_sha256"] = historical_scene
                compact["source_objective_instance_sha256"] = historical_objective
                metadata["static_first_step_newton_reference"] = compact
                metadata["static_first_step_newton_reference"]["dense"][field] = value
                path = _write_unvalidated_bundle(pathlib.Path(directory), metadata, arrays)
                with self.assertRaisesRegex(ValueError, "inverted|non-positive|exact pins"):
                    _load_test_bundle(path)

    def test_wrong_external_file_digest_fails_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path, _json_sha256, npz_sha256 = self._write_bundle(directory)
            with self.assertRaisesRegex(ValueError, "JSON file digest"):
                recording.load_authenticated_medium_reference(
                    json_path,
                    expected_json_sha256="0" * 64,
                    expected_npz_sha256=npz_sha256,
                    npz_path=npz_path,
                )

    def test_npz_override_cannot_be_silently_ignored_without_reference_json(self):
        with self.assertRaises(SystemExit):
            recording.main(
                [
                    "generate",
                    "--scene",
                    "refinement-medium",
                    "--frames",
                    "1",
                    "--out-dir",
                    "/tmp/not-used",
                    "--static-reference-npz",
                    "/tmp/not-used.npz",
                ]
            )


class TestRenderPolicy(unittest.TestCase):
    def test_default_is_realtime_and_explicit_retiming_is_visible(self):
        metadata, _arrays = _valid_bundle()
        self.assertEqual(recording._resolve_render_fps(metadata, None), (60, 1.0))
        self.assertEqual(recording._resolve_render_fps(metadata, 30), (30, 0.5))
        self.assertEqual(recording._playback_annotation(1.0), "")
        self.assertEqual(recording._playback_annotation(0.5), "playback 0.5x slow motion")
        self.assertEqual(recording._playback_annotation(2.0), "playback 2x fast motion")
        args = recording._build_parser().parse_args(["render", "--bundle", "bundle.json", "--out", "proof.mp4"])
        self.assertIsNone(args.fps)

    def test_render_sidecar_binds_bundle_mp4_timing_and_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            metadata, arrays = _valid_bundle()
            bundle = _save_validated_bundle(directory, metadata, arrays)
            bundle_record = json.loads(bundle.read_text(encoding="utf-8"))
            output = root / "proof.mp4"
            output.write_bytes(b"mp4 bytes")
            encoded_stream = {"codec": "h264", "pixel_format": "yuv420p", "encoded_duration_seconds": 2.0 / 60.0}
            record = recording._build_render_record(
                bundle_json=bundle,
                bundle_record_sha256=bundle_record["record_sha256"],
                output_mp4=output,
                fps=60,
                source_fps=60,
                frame_count=2,
                width=1920,
                height=720,
                render_source=_fake_render_source(),
                encoded_stream=encoded_stream,
            )
            sidecar = recording._save_render_record(recording._render_sidecar_path(output), record)
            with (
                mock.patch.object(recording, "_inspect_encoded_mp4", return_value=encoded_stream),
                mock.patch.object(recording, "_verify_generation_source_git_objects"),
            ):
                loaded = recording.load_render_record(sidecar)
            self.assertEqual(loaded["bundle_record_sha256"], bundle_record["record_sha256"])
            self.assertEqual(loaded["playback_rate"], 1.0)
            self.assertEqual((loaded["fps"], loaded["frame_count"]), (60, 2))
            self.assertEqual((loaded["width"], loaded["height"]), (1920, 720))
            self.assertEqual(
                (loaded["encoder"], loaded["codec"], loaded["pixel_format"]), ("libx264", "h264", "yuv420p")
            )

            output.write_bytes(b"tampered mp4 bytes")
            with (
                mock.patch.object(recording, "_verify_generation_source_git_objects"),
                self.assertRaisesRegex(ValueError, "MP4 bytes"),
            ):
                recording.load_render_record(sidecar)

    @unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg"), "optional imageio-ffmpeg is unavailable")
    def test_encoded_stream_probe_checks_actual_frames_rate_dimensions_and_codec(self):
        import imageio_ffmpeg  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "probe.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(output),
                (16, 16),
                fps=60,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=16,
            )
            writer.send(None)
            writer.send(bytes(16 * 16 * 3))
            writer.send(bytes([127]) * (16 * 16 * 3))
            writer.close()
            stream = recording._inspect_encoded_mp4(
                output,
                expected_fps=60,
                expected_frame_count=2,
                expected_width=16,
                expected_height=16,
            )
            self.assertEqual((stream["codec"], stream["pixel_format"]), ("h264", "yuv420p"))
            self.assertGreater(stream["encoded_duration_seconds"], 0.0)
            with self.assertRaisesRegex(RuntimeError, "frames"):
                recording._inspect_encoded_mp4(
                    output,
                    expected_fps=60,
                    expected_frame_count=3,
                    expected_width=16,
                    expected_height=16,
                )

    def test_render_record_rejects_odd_dimensions_and_forged_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bundle = root / "bundle.json"
            output = root / "proof.mp4"
            bundle.write_text(json.dumps({"record_sha256": "4" * 64}), encoding="utf-8")
            output.write_bytes(b"mp4 bytes")
            record = recording._build_render_record(
                bundle_json=bundle,
                bundle_record_sha256="4" * 64,
                output_mp4=output,
                fps=30,
                source_fps=60,
                frame_count=61,
                width=1919,
                height=720,
                render_source=_fake_render_source(),
                encoded_stream={"codec": "h264", "pixel_format": "yuv420p", "encoded_duration_seconds": 2.03},
            )
            with self.assertRaisesRegex(ValueError, "dimensions"):
                recording._validate_render_record(record)
            record["width"] = 1920
            del record["render_record_sha256"]
            record = _seal_manifest(record, "render_record_sha256")
            recording._validate_render_record(record)
            wrong_duration = copy.deepcopy(record)
            wrong_duration["encoded_duration_seconds"] = 999.0
            del wrong_duration["render_record_sha256"]
            wrong_duration = _seal_manifest(wrong_duration, "render_record_sha256")
            with self.assertRaisesRegex(ValueError, "duration"):
                recording._validate_render_record(wrong_duration)
            record["render_source"]["files"]["newton_capture/_video.py"] = "0" * 64
            del record["render_record_sha256"]
            record = _seal_manifest(record, "render_record_sha256")
            with self.assertRaisesRegex(ValueError, "render source"):
                recording._validate_render_record(record)


class TestRecordingOverlay(unittest.TestCase):
    def test_three_panel_overlay_retains_rgb_shape(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        methods = recording._method_records(recording.recording_spec("twist"))
        record = {
            "methods": methods,
            "scene_display_name": "unit-test twist",
        }
        arrays = {f"metric_{name}": np.full((3, 2), np.nan) for name in recording.METRIC_NAMES}
        arrays.update(
            {
                "mg_last_gate_accepted": np.array([[-1] * 4, [1, 1, 0, 1]], dtype=np.int8),
                "source_frame_index": np.array([-1, 0], dtype=np.int64),
                "time_seconds": np.array([0.0, 1.0 / 60.0]),
            }
        )
        arrays["metric_relative_residual"][:, 1] = [1.0e-7, 2.0e-4, 3.0e-3]
        arrays["metric_free_rms_error_m"][:, 1] = [0.0, 1.0e-5, 2.0e-5]
        arrays["metric_determinant_min"][:, 1] = [0.99, 0.98, 0.97]
        arrays["metric_inverted_tet_fraction"][:, 1] = 0.0
        arrays["metric_max_pin_error_m"][:, 1] = 0.0
        panels = [
            recording.label_panel(
                frame,
                method_index=index,
                frame_index=1,
                record=record,
                arrays=arrays,
            )
            for index in range(3)
        ]
        composite = recording.label_composite(
            np.concatenate(panels, axis=1),
            frame_index=1,
            record=record,
            arrays=arrays,
        )
        self.assertEqual(composite.shape, (240, 960, 3))
        self.assertEqual(composite.dtype, np.uint8)
        self.assertGreater(int(composite.sum()), 0)


if __name__ == "__main__":
    unittest.main()
