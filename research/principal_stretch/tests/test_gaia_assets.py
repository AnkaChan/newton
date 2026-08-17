# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for audited Gaia tetrahedral benchmark assets."""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

import numpy as np

import newton

from ..gaia_assets import GAIA_ASSETS, build_gaia_tet_scene, build_registered_gaia_scene, load_gaia_tet_mesh

_INVERTED_FIXTURE = """\
# Sparse, out-of-order source IDs with one deliberately inverted tet.
Vertex 40 0 0 1
Vertex 10 0 0 0
Vertex 30 0 1 0
Vertex 20 1 0 0
Tet 7 10 30 20 40
"""


class TestGaiaAssets(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture_path = pathlib.Path(self.temporary_directory.name) / "fixture.t"
        self.fixture_path.write_text(_INVERTED_FIXTURE, encoding="utf-8")

    def test_parser_maps_explicit_ids_scales_and_repairs_orientation(self):
        mesh = load_gaia_tet_mesh(self.fixture_path, unit_scale_m_per_source_unit=0.25)
        np.testing.assert_array_equal(mesh.source_vertex_ids, [10, 20, 30, 40])
        np.testing.assert_array_equal(mesh.source_tet_ids, [7])
        np.testing.assert_array_equal(mesh.tet_indices, [[0, 1, 2, 3]])
        np.testing.assert_array_equal(mesh.vertices_m[1], [0.25, 0.0, 0.0])
        self.assertEqual(mesh.original_inverted_count, 1)
        self.assertEqual(mesh.orientation_repaired_count, 1)
        np.testing.assert_array_equal(mesh.repaired_source_tet_ids, [7])
        self.assertGreater(mesh.signed_six_volumes_m3[0], 0.0)
        with self.assertRaises(ValueError):
            mesh.vertices_m[0, 0] = 1.0

    def test_parser_can_preserve_a_valid_negative_orientation(self):
        mesh = load_gaia_tet_mesh(
            self.fixture_path,
            unit_scale_m_per_source_unit=1.0,
            repair_orientation=False,
        )
        self.assertEqual(mesh.original_inverted_count, 1)
        self.assertEqual(mesh.orientation_repaired_count, 0)
        self.assertLess(mesh.signed_six_volumes_m3[0], 0.0)

    def test_hermetic_fixture_builds_public_newton_scene_with_bound_hashes(self):
        scene = build_gaia_tet_scene(
            self.fixture_path,
            name="gaia-hermetic-fixture",
            source_revision="fixture-revision",
            source_relative_path="fixtures/fixture.t",
            unit_scale_m_per_source_unit=0.5,
            density=24.0,
            mu=10.0,
            public_lambda=20.0,
            support_axis=0,
            boundary_fraction=0.0,
            gravity=(0.0, 0.0, -2.0),
            total_tip_force=(3.0, -2.0, 1.0),
            dt=0.125,
        )
        self.assertEqual((scene.n_vertices, scene.n_tets, scene.n_triangles), (4, 1, 4))
        np.testing.assert_array_equal(scene.pinned_indices, [0, 2, 3])
        self.assertTrue(np.all((scene.particle_flags[scene.pinned_indices] & int(newton.ParticleFlags.ACTIVE)) == 0))
        np.testing.assert_array_equal(scene.external_force.sum(axis=0), [3.0, -2.0, 1.0])
        np.testing.assert_array_equal(scene.tet_materials, [[10.0, 30.0, 0.0]])
        self.assertEqual(scene.metadata["orientation_repaired_tet_count"], 1)
        self.assertEqual(scene.metadata["unit_scale_m_per_source_unit"], 0.5)
        for key in (
            "source_file_sha256",
            "source_topology_sha256",
            "geometry_sha256",
            "topology_sha256",
            "material_sha256",
            "boundary_sha256",
        ):
            self.assertEqual(len(scene.metadata[key]), 64)
        manifest = scene.manifest()
        self.assertEqual(manifest["metadata"]["source_revision"], "fixture-revision")
        self.assertEqual(len(manifest["scene_sha256"]), 64)

    def test_parser_rejects_malformed_or_unauditable_meshes(self):
        invalid_cases = (
            ("unknown", "Vertex or Tet record", "Point 0 0 0 0\nTet 0 0 1 2 3\n"),
            ("missing", "missing Vertex", "Vertex 0 0 0 0\nVertex 1 1 0 0\nVertex 2 0 1 0\nTet 0 0 1 2 3\n"),
            (
                "nonfinite",
                "invalid Vertex values",
                "Vertex 0 nan 0 0\nVertex 1 1 0 0\nVertex 2 0 1 0\nVertex 3 0 0 1\nTet 0 0 1 2 3\n",
            ),
            (
                "degenerate",
                "degenerate",
                "Vertex 0 0 0 0\nVertex 1 1 0 0\nVertex 2 0 1 0\nVertex 3 1 1 0\nTet 0 0 1 2 3\n",
            ),
        )
        for filename, expected, contents in invalid_cases:
            with self.subTest(expected=expected):
                path = pathlib.Path(self.temporary_directory.name) / f"{filename}.t"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    load_gaia_tet_mesh(path, unit_scale_m_per_source_unit=1.0)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_gaia_tet_mesh(
                self.fixture_path,
                unit_scale_m_per_source_unit=1.0,
                expected_file_sha256="0" * 64,
            )


@unittest.skipUnless(os.environ.get("PSS_GAIA_ASSET_ROOT"), "set PSS_GAIA_ASSET_ROOT for Gaia asset tests")
class TestGaiaRealAssets(unittest.TestCase):
    def test_primary_assets_build_and_match_pinned_digests(self):
        root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        expected = {
            "bunny_small": (1839, 5891, 0.1),
            "Armadilo_lowres": (3992, 14870, 1.0),
        }
        for name, (n_vertices, n_tets, scale) in expected.items():
            with self.subTest(asset=name):
                scene = build_registered_gaia_scene(
                    name,
                    root,
                    unit_scale_m_per_source_unit=scale,
                    total_tip_force=(10.0, 0.0, 0.0),
                )
                self.assertEqual((scene.n_vertices, scene.n_tets), (n_vertices, n_tets))
                self.assertEqual(scene.metadata["source_file_sha256"], GAIA_ASSETS[name].sha256)
                self.assertGreater(scene.pinned_indices.size, 0)
                self.assertGreater(scene.free_indices.size, 0)
                np.testing.assert_allclose(scene.external_force.sum(axis=0), [10.0, 0.0, 0.0], atol=2.0e-5)

    def test_spaghetti_parser_is_a_high_resolution_sliver_smoke(self):
        root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        spec = GAIA_ASSETS["spaghetti"]
        mesh = load_gaia_tet_mesh(
            root / spec.relative_path,
            unit_scale_m_per_source_unit=0.1,
            expected_file_sha256=spec.sha256,
        )
        self.assertEqual((mesh.vertices_m.shape[0], mesh.tet_indices.shape[0]), (28776, 96869))
        self.assertEqual(mesh.orientation_repaired_count, 0)
        self.assertGreater(mesh.signed_six_volumes_m3.min(), 0.0)
        extent = np.ptp(mesh.vertices_m, axis=0)
        self.assertGreater(extent.max() / np.partition(extent, 1)[1], 1000.0)


if __name__ == "__main__":
    unittest.main()
