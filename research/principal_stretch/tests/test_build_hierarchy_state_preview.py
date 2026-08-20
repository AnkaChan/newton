# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the static hierarchy/state preview builder."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import struct
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

from .. import build_hierarchy_state_preview as preview_builder
from ..build_hierarchy_state_preview import (
    _ZIP_TIMESTAMP,
    ASSET_BASENAMES,
    _parse_args,
    _sparse_vectors,
    build_preview,
    default_asset_paths,
    load_legacy_vtk_tet_mesh,
)

_TWO_TET_VTK = """\
# vtk DataFile Version 3.0
two tetrahedra
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 6 double
0 0 0  9 9
9  1 0 0  0 1 0
0 0 1
0 0 -1
CELLS 2 10
4 0 2 3 4
4 0 3 2 5
CELL_TYPES 2
10 10
"""


class TestLegacyVTKTetParser(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = pathlib.Path(self.temporary_directory.name)
        self.fixture = self.root / "two_tets.vtk"
        self.fixture.write_text(_TWO_TET_VTK, encoding="ascii")

    def test_parser_reads_wrapped_points_and_tetrahedra(self) -> None:
        mesh = load_legacy_vtk_tet_mesh(self.fixture)
        np.testing.assert_array_equal(
            mesh.rest_positions,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        )
        np.testing.assert_array_equal(mesh.tet_indices, [[0, 1, 2, 3], [0, 2, 1, 4]])
        self.assertEqual(mesh.source_point_count, 6)
        self.assertEqual(mesh.dropped_unused_point_count, 1)
        self.assertEqual(mesh.source_sha256, hashlib.sha256(self.fixture.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            mesh.rest_positions[0, 0] = 2.0

    def test_parser_rejects_binary_non_tet_malformed_and_over_cap_inputs(self) -> None:
        cases = {
            "binary": (
                "only legacy ASCII",
                _TWO_TET_VTK.replace("ASCII", "BINARY", 1),
                {},
            ),
            "non_tet": (
                "non-tetrahedral",
                _TWO_TET_VTK.replace("CELLS 2 10\n4 0 2 3 4\n4 0 3 2 5", "CELLS 1 4\n3 0 1 2").replace(
                    "CELL_TYPES 2\n10 10", "CELL_TYPES 1\n5"
                ),
                {},
            ),
            "malformed": (
                "truncated",
                _TWO_TET_VTK.rsplit("10", 1)[0],
                {},
            ),
            "over_cap": (
                "exceeds cap",
                _TWO_TET_VTK,
                {"max_tets": 1},
            ),
            "bad_index": (
                "outside the POINTS range",
                _TWO_TET_VTK.replace("4 0 3 2 5", "4 0 3 2 7"),
                {},
            ),
        }
        for name, (message, contents, kwargs) in cases.items():
            with self.subTest(name=name):
                path = self.root / f"{name}.vtk"
                path.write_text(contents, encoding="ascii")
                with self.assertRaisesRegex(ValueError, message):
                    load_legacy_vtk_tet_mesh(path, **kwargs)


class TestHierarchyStatePreview(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = pathlib.Path(self.temporary_directory.name)
        self.fixture = self.root / "fixture.vtk"
        self.fixture.write_text(_TWO_TET_VTK, encoding="ascii")

    def test_default_inventory_has_exact_five_basenames(self) -> None:
        asset_root = self.root / "assets"
        expected = tuple(asset_root / f"{name}.vtk" for name in ASSET_BASENAMES)
        self.assertEqual(default_asset_paths(asset_root), expected)
        with mock.patch.dict(os.environ, {"PSS_VTK_ASSET_DIR": str(asset_root)}):
            self.assertEqual(default_asset_paths(), expected)
        default_args = _parse_args(["--output-dir", str(self.root / "output")])
        self.assertEqual(default_args.n_levels, 5)
        self.assertEqual(default_args.samples_per_asset, 1)
        self.assertEqual(
            _parse_args(["--output-dir", str(self.root / "output"), "--samples-per-asset", "3"]).samples_per_asset,
            3,
        )

    def test_sparse_vectors_evenly_subsamples_only_boundary_candidates(self) -> None:
        vectors = np.ones((160, 3), dtype=np.float64)
        boundary_vertices = np.arange(1, 160, 2, dtype=np.int64)
        selected = _sparse_vectors(vectors, candidate_indices=boundary_vertices)
        expected = boundary_vertices[np.arange(72, dtype=np.int64) * boundary_vertices.size // 72]
        np.testing.assert_array_equal(selected, expected)

    def test_multi_sample_gallery_is_distinct_deterministic_and_builds_hierarchy_once(self) -> None:
        first_output = self.root / "multi_first"
        second_output = self.root / "multi_second"
        with mock.patch.object(
            preview_builder,
            "build_hierarchy",
            wraps=preview_builder.build_hierarchy,
        ) as build_hierarchy_mock:
            first = build_preview(
                first_output,
                asset_paths=[self.fixture],
                base_seed=17,
                cluster_size=2,
                samples_per_asset=3,
            )
        self.assertEqual(build_hierarchy_mock.call_count, 1)
        second = build_preview(
            second_output,
            asset_paths=[self.fixture],
            base_seed=17,
            cluster_size=2,
            samples_per_asset=3,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["samples_per_asset"], 3)
        asset = first["assets"][0]
        samples = asset["samples"]
        self.assertEqual([sample["sample_index"] for sample in samples], [0, 1, 2])
        seed_pairs = [(sample["deformation_seed"], sample["velocity_seed"]) for sample in samples]
        self.assertEqual(len(set(seed_pairs)), 3)
        self.assertEqual(len({seed for pair in seed_pairs for seed in pair}), 6)
        self.assertEqual(asset["deformation_seed"], samples[0]["deformation_seed"])
        self.assertEqual(asset["velocity_seed"], samples[0]["velocity_seed"])
        self.assertEqual(asset["metrics"], samples[0]["metrics"])
        self.assertEqual(asset["outputs"]["state_png"], samples[0]["outputs"]["state_png"])
        self.assertEqual(asset["outputs"]["state_npz"], samples[0]["outputs"]["state_npz"])

        source_sha256 = hashlib.sha256(self.fixture.read_bytes()).hexdigest()

        def expected_seed(sample_index: int, role: str) -> int:
            prefix = f"17:fixture:{source_sha256}"
            if sample_index:
                prefix = f"{prefix}:sample:{sample_index}"
            return int.from_bytes(hashlib.sha256(f"{prefix}:{role}".encode("ascii")).digest()[:4], "little")

        for sample in samples:
            sample_index = sample["sample_index"]
            self.assertTrue(
                {"sample_index", "deformation_seed", "velocity_seed", "metrics", "outputs"} <= sample.keys()
            )
            self.assertEqual(sample["deformation_seed"], expected_seed(sample_index, "deformation"))
            self.assertEqual(sample["velocity_seed"], expected_seed(sample_index, "velocity"))
            level_scales = sample["metrics"]["deformation_level_scales"]
            effective_scales = sample["metrics"]["effective_deformation_level_scales"]
            uniform_scale = sample["metrics"]["deformation_scale"]
            self.assertEqual(len(level_scales), len(asset["hierarchy"]["levels"]))
            self.assertTrue(all(type(scale) is float and 0.0 < scale <= 1.0 for scale in level_scales))
            np.testing.assert_allclose(effective_scales, np.asarray(level_scales) * uniform_scale)
            self.assertEqual(
                sample["state_figure"]["original_surface_material"],
                sample["state_figure"]["after_surface_material"],
            )

        camera_bounds = [sample["state_figure"]["camera_bounds"] for sample in samples]
        self.assertEqual(camera_bounds, [camera_bounds[0]] * 3)
        expected_names = {
            "fixture_hierarchy.png",
            "fixture_state.png",
            "fixture_state.npz",
            "fixture_state_sample_1.png",
            "fixture_state_sample_1.npz",
            "fixture_state_sample_2.png",
            "fixture_state_sample_2.npz",
            "index.html",
            "manifest.json",
        }
        self.assertEqual({path.name for path in first_output.iterdir()}, expected_names)
        self.assertEqual({record["path"] for record in first["artifact_inventory"]}, expected_names - {"manifest.json"})
        for name in expected_names:
            self.assertEqual((first_output / name).read_bytes(), (second_output / name).read_bytes())

        html_text = (first_output / "index.html").read_text(encoding="utf-8")
        self.assertIn('class="sample-gallery" data-sample-count="3"', html_text)
        self.assertEqual(html_text.count('class="state-strip"'), 3)
        for sample_index in range(3):
            self.assertIn(f'data-sample-index="{sample_index}"', html_text)
        manifest_text = (first_output / "manifest.json").read_text(encoding="utf-8").lower()
        render_source = inspect.getsource(preview_builder._render_state_image).lower()
        for forbidden_rendering in ("heatmap", "magma", "colorbar"):
            self.assertNotIn(forbidden_rendering, render_source)
            self.assertNotIn(forbidden_rendering, html_text.lower())
            self.assertNotIn(forbidden_rendering, manifest_text)

    def test_samples_per_asset_requires_positive_builtin_int(self) -> None:
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "samples_per_asset must be a positive integer"):
                    build_preview(self.root / "invalid", asset_paths=[self.fixture], samples_per_asset=invalid)

    def test_builder_writes_referenced_static_inventory_and_deterministic_manifest(self) -> None:
        first_output = self.root / "first"
        second_output = self.root / "second"
        first = build_preview(first_output, asset_paths=[self.fixture], base_seed=17, cluster_size=2)
        second = build_preview(second_output, asset_paths=[self.fixture], base_seed=17, cluster_size=2)

        expected_names = {
            "fixture_hierarchy.png",
            "fixture_state.png",
            "fixture_state.npz",
            "index.html",
            "manifest.json",
        }
        self.assertEqual({path.name for path in first_output.iterdir()}, expected_names)
        self.assertEqual((first_output / "manifest.json").read_bytes(), (second_output / "manifest.json").read_bytes())
        self.assertEqual(
            (first_output / "fixture_state.npz").read_bytes(), (second_output / "fixture_state.npz").read_bytes()
        )
        self.assertEqual(json.loads((first_output / "manifest.json").read_text(encoding="utf-8")), first)
        self.assertEqual(first, second)
        self.assertFalse(first["is_dynamics"])
        self.assertIn("not dynamics", first["notice"])
        self.assertEqual(first["samples_per_asset"], 1)
        self.assertEqual(first["hierarchy_config"]["n_levels"], 5)
        self.assertEqual(first["state_config"]["level_decay"], 0.70)
        asset = first["assets"][0]
        self.assertEqual(len(asset["samples"]), 1)
        self.assertEqual(asset["samples"][0]["sample_index"], 0)
        self.assertEqual(asset["point_count"], 5)
        self.assertEqual(asset["source_point_count"], 6)
        self.assertEqual(asset["dropped_unused_point_count"], 1)
        self.assertEqual(asset["source_sha256"], hashlib.sha256(self.fixture.read_bytes()).hexdigest())

        html_text = (first_output / "index.html").read_text(encoding="utf-8")
        for reference in ("fixture_hierarchy.png", "fixture_state.png", "fixture_state.npz", "manifest.json"):
            self.assertIn(reference, html_text)
        self.assertIn("5 active vertices (1 unused source point dropped)", html_text)
        self.assertIn("Asset size means the diagonal of the original mesh bounding box", html_text)
        self.assertIn("Maximum movement", html_text)
        self.assertIn("% of asset size", html_text)
        self.assertIn("Maximum speed", html_text)
        self.assertIn("% of asset size per second", html_text)
        self.assertIn("smallest local volume ratio", html_text)
        self.assertIn("smallest directional stretch", html_text)
        self.assertIn("safety threshold 0.35", html_text)
        self.assertIn("Per-level safety multipliers, from fine to coarsest", html_text)
        self.assertIn("final uniform cap/validity multiplier", html_text)
        self.assertIn("resulting effective multipliers", html_text)
        self.assertNotIn("sampled deformation stayed at 100% strength", html_text.lower())
        self.assertIn("<details>", html_text)
        self.assertNotIn("minimum determinant", html_text)
        self.assertNotIn("minimum singular value", html_text)
        self.assertNotIn("characteristic length", html_text)
        self.assertNotIn("/L", html_text)
        self.assertIn(
            "velocity arrows uniformly rescaled for visibility; directions and relative lengths retained",
            html_text,
        )
        self.assertNotIn("<script", html_text.lower())
        hierarchy_png = (first_output / "fixture_hierarchy.png").read_bytes()
        state_png = (first_output / "fixture_state.png").read_bytes()
        self.assertTrue(hierarchy_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(state_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", hierarchy_png[16:24]), (1066, 546))
        self.assertEqual(struct.unpack(">II", state_png[16:24]), (2080, 572))
        self.assertGreater(asset["state_figure"]["velocity_display_scale_factor"], 0.0)
        self.assertEqual(asset["state_figure"]["surface_style"], "per_face_depth_shading_with_visible_edges")
        self.assertEqual(
            asset["state_figure"]["velocity_reference_surface"],
            "translucent_exact_deformed_reference_surface",
        )
        self.assertEqual(asset["state_figure"]["velocity_reference_surface_alpha"], 0.38)
        self.assertEqual(asset["state_figure"]["velocity_reference_edge_alpha"], 0.10)
        self.assertIn("translucent exact-deformed reference surface", html_text)
        self.assertEqual(asset["state_figure"]["arrow_candidate_scope"], "boundary_surface_vertices")
        self.assertEqual(asset["state_figure"]["boundary_arrow_candidate_count"], 5)
        render_source = inspect.getsource(preview_builder._render_state_image).lower()
        for forbidden_rendering in ("heatmap", "magma", "colorbar"):
            self.assertNotIn(forbidden_rendering, render_source)
            self.assertNotIn(forbidden_rendering, html_text.lower())
        self.assertEqual(
            asset["state_figure"]["original_surface_material"],
            asset["state_figure"]["after_surface_material"],
        )
        self.assertEqual(asset["state_figure"]["surface_color_mode"], "uniform_solid_material")
        self.assertEqual(asset["state_figure"]["overlay_surface_color_mode"], "uniform_solid_material")
        self.assertNotEqual(
            asset["state_figure"]["overlay_deformed_surface_material"],
            asset["state_figure"]["original_ghost_surface_material"],
        )
        self.assertEqual(asset["state_figure"]["after_geometry"], "exact_deformed_positions")
        self.assertEqual(asset["state_figure"]["headline"], "Generated initial state \N{EM DASH} not a simulation")
        self.assertNotIn("after_displacement_display_magnification", asset["state_figure"])
        self.assertNotIn("after_displacement_display_is_presentation_only", asset["state_figure"])
        self.assertNotIn("magnification", html_text.lower())
        self.assertIn("same uniform solid material, identical camera and bounds", html_text)
        self.assertGreaterEqual(html_text.count('class="figure-scroll"'), 2)
        self.assertIn(".hierarchy-strip { width: auto; max-width: none; }", html_text)
        self.assertIn(".state-strip { width: auto; max-width: none; }", html_text)
        self.assertNotIn(".hierarchy-strip { width: 1500px", html_text)
        self.assertNotIn(".state-strip { width: 1200px", html_text)
        self.assertIn(
            '<p class="scroll-hint">Swipe horizontally to see all hierarchy levels &rarr;</p><div class="figure-scroll">',
            html_text,
        )
        self.assertIn(
            '<p class="scroll-hint">Swipe horizontally to see all four state panels &rarr;</p><div class="figure-scroll">',
            html_text,
        )
        self.assertIn(".scroll-hint { display: none;", html_text)
        self.assertIn(".scroll-hint { display: block;", html_text)
        state_record = first["assets"][0]["outputs"]["state_npz"]
        self.assertEqual(
            set(state_record["arrays"]),
            {"rest_positions", "deformed_positions", "velocities", "tet_indices"},
        )
        with zipfile.ZipFile(first_output / "fixture_state.npz") as archive:
            self.assertTrue(all(info.date_time == _ZIP_TIMESTAMP for info in archive.infolist()))
        with np.load(first_output / "fixture_state.npz", allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), set(state_record["arrays"]))
            np.testing.assert_array_equal(arrays["tet_indices"], [[0, 1, 2, 3], [0, 2, 1, 4]])
            self.assertEqual(arrays["deformed_positions"].shape, (5, 3))
            self.assertEqual(arrays["velocities"].shape, (5, 3))
            stored_max_displacement = float(
                np.linalg.norm(arrays["deformed_positions"] - arrays["rest_positions"], axis=1).max()
            )
            self.assertAlmostEqual(stored_max_displacement, asset["metrics"]["max_displacement_asset_units"])


if __name__ == "__main__":
    unittest.main()
