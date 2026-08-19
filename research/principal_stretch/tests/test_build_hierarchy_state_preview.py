# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the static hierarchy/state preview builder."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

from ..build_hierarchy_state_preview import (
    _ZIP_TIMESTAMP,
    ASSET_BASENAMES,
    _parse_args,
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
        self.assertEqual(_parse_args(["--output-dir", str(self.root / "output")]).n_levels, 5)

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
        self.assertEqual(first["hierarchy_config"]["n_levels"], 5)
        self.assertEqual(first["state_config"]["level_decay"], 0.10)
        asset = first["assets"][0]
        self.assertEqual(asset["point_count"], 5)
        self.assertEqual(asset["source_point_count"], 6)
        self.assertEqual(asset["dropped_unused_point_count"], 1)
        self.assertEqual(asset["source_sha256"], hashlib.sha256(self.fixture.read_bytes()).hexdigest())
        self.assertIn("max_centered_displacement_fraction", asset["metrics"])

        html_text = (first_output / "index.html").read_text(encoding="utf-8")
        for reference in ("fixture_hierarchy.png", "fixture_state.png", "fixture_state.npz", "manifest.json"):
            self.assertIn(reference, html_text)
        self.assertIn("5 active vertices (1 unused source point dropped)", html_text)
        self.assertIn("max centered displacement/L (mean translation removed)", html_text)
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


if __name__ == "__main__":
    unittest.main()
