# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the audited multi-scene baseline runner."""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from .. import run_solver_scenes as runner


class TestRunSolverScenes(unittest.TestCase):
    def test_scene_aliases_expand_deterministically(self):
        self.assertEqual(
            runner._resolve_scene_keys(("stretch", "refinement", "stretch")),
            ("stretch", "refinement-coarse", "refinement-medium", "refinement-fine"),
        )
        self.assertEqual(runner._resolve_scene_keys(("all",)), tuple(runner._SCENE_FACTORIES))
        with self.assertRaisesRegex(ValueError, "unknown scene"):
            runner._resolve_scene_keys(("not-a-scene",))

    def test_vbd_budgets_are_canonical_and_must_be_distinct(self):
        self.assertEqual(runner._normalize_vbd_iterations((8, 1, 4)), (1, 4, 8))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            runner._normalize_vbd_iterations((0,))
        with self.assertRaisesRegex(ValueError, "distinct"):
            runner._normalize_vbd_iterations((2, 2))

    def test_dense_ceiling_fails_before_solver_or_output(self):
        scene = types.SimpleNamespace(
            free_indices=np.arange(2),
            n_vertices=3,
            n_tets=1,
            name="oversized",
        )
        factory = mock.Mock(return_value=scene)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "not-created"
            config = runner.SolverSceneRunConfig(
                output_dir=output,
                scene_selectors=("extension",),
                max_newton_free_dofs=5,
            )
            with (
                mock.patch.dict(runner._SCENE_FACTORIES, {"extension": factory}, clear=True),
                mock.patch.object(runner, "build_common_problem") as build_problem,
                mock.patch.object(runner, "run_newton") as run_newton,
            ):
                with self.assertRaisesRegex(ValueError, "free-DOF ceiling 5 exceeded"):
                    runner.run_solver_scenes(config)
            factory.assert_called_once_with()
            build_problem.assert_not_called()
            run_newton.assert_not_called()
            self.assertFalse(output.exists())

    def test_runner_writes_bound_index_and_restarts_each_budget(self):
        scene_hash = "a" * 64

        class FakeScene:
            name = "audited-default"
            n_vertices = 4
            n_tets = 1
            free_indices = np.array([1, 2], dtype=np.int64)

            @staticmethod
            def manifest():
                return {"scene_sha256": scene_hash}

        scene = FakeScene()
        problem = object()
        newton_result = types.SimpleNamespace(reference_accepted=True, reference_failures=())

        def write_bundle(path, supplied_scene, supplied_problem, supplied_newton, vbd_results):
            self.assertIs(supplied_scene, scene)
            self.assertIs(supplied_problem, problem)
            self.assertIs(supplied_newton, newton_result)
            self.assertEqual(vbd_results, ["vbd-1", "vbd-4"])
            raw_path = path.with_suffix(".npz")
            raw_path.write_bytes(b"raw-state")
            raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            payload = {
                "scene": {"scene_sha256": scene_hash},
                "raw_npz": {"path": raw_path.name, "sha256": raw_sha256},
            }
            path.write_text(json.dumps(payload))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "results"
            config = runner.SolverSceneRunConfig(
                output_dir=output,
                scene_selectors=("extension",),
                device="cuda:0",
                tile_solve=True,
                repeats=3,
                vbd_iterations=(4, 1),
                max_newton_free_dofs=6,
            )
            with (
                mock.patch.dict(runner._SCENE_FACTORIES, {"extension": mock.Mock(return_value=scene)}, clear=True),
                mock.patch.object(runner, "build_common_problem", return_value=problem) as build_problem,
                mock.patch.object(runner, "run_newton", return_value=newton_result) as run_newton,
                mock.patch.object(
                    runner, "run_vbd", side_effect=lambda _scene, budget, **_kwargs: f"vbd-{budget}"
                ) as run_vbd,
                mock.patch.object(runner, "write_benchmark_bundle", side_effect=write_bundle) as write,
            ):
                index_path = runner.run_solver_scenes(config)

            self.assertEqual(index_path, output.resolve() / "index.json")
            build_problem.assert_called_once_with(scene)
            run_newton.assert_called_once_with(scene, problem, warmup=True, repeats=3)
            self.assertEqual([call.args[1] for call in run_vbd.call_args_list], [1, 4])
            for call in run_vbd.call_args_list:
                self.assertIs(call.args[0], scene)
                self.assertEqual(
                    call.kwargs,
                    {"device": "cuda:0", "tile_solve": True, "warmup": True, "repeats": 3},
                )
            write.assert_called_once()

            index = json.loads(index_path.read_text())
            stored_digest = index.pop("index_sha256")
            self.assertEqual(stored_digest, runner._json_sha256(index))
            self.assertEqual(index["configuration"]["scene_parameters"], "audited builder defaults")
            self.assertEqual(index["configuration"]["vbd_iterations"], [1, 4])
            self.assertEqual(index["scenes"][0]["scene_sha256"], scene_hash)
            bundle = output / index["scenes"][0]["bundle"]["path"]
            raw = output / index["scenes"][0]["raw_npz"]["path"]
            self.assertEqual(index["scenes"][0]["bundle"]["sha256"], runner._file_sha256(bundle))
            self.assertEqual(index["scenes"][0]["raw_npz"]["sha256"], runner._file_sha256(raw))

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory)
            (output / "extension.json").write_text("owned")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                runner._assert_output_paths_available(output, ("extension",))

    def test_rejected_newton_reference_stops_before_vbd(self):
        scene = types.SimpleNamespace(
            free_indices=np.arange(1),
            n_vertices=2,
            n_tets=1,
            name="rejected",
        )
        rejected = types.SimpleNamespace(reference_accepted=False, reference_failures=("inverted",))
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = runner.SolverSceneRunConfig(
                output_dir=pathlib.Path(temporary_directory) / "results",
                scene_selectors=("extension",),
                max_newton_free_dofs=3,
            )
            with (
                mock.patch.dict(
                    runner._SCENE_FACTORIES,
                    {"extension": mock.Mock(return_value=scene)},
                    clear=True,
                ),
                mock.patch.object(runner, "build_common_problem", return_value=object()),
                mock.patch.object(runner, "run_newton", return_value=rejected),
                mock.patch.object(runner, "run_vbd") as run_vbd,
                mock.patch.object(runner, "write_benchmark_bundle") as write,
            ):
                with self.assertRaisesRegex(RuntimeError, "Newton reference gates failed"):
                    runner.run_solver_scenes(config)
            run_vbd.assert_not_called()
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
